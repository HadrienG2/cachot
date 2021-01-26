//! Representation of previously visited points on a PartialPath

use super::FeedPair;
use crate::cache;
use slotmap::{DefaultKey, SlotMap};

/// Storage for PartialPath path elements
///
/// PartialPath steps are stored as a linked list, with node deduplication so
/// that when a path forks into multiple sub-paths, we don't need to make
/// multiple copies of the parent path (which costs RAM capacity and bandwidth).
///
/// Originally, this list was stored using Rc<PathElem>, but the overhead
/// associated with allocating and liberating all those PathElems turned out to
/// be too great. So we're now reusing allocations instead.
///
pub struct PathElemStorage(SlotMap<DefaultKey, PathElem>);
//
impl PathElemStorage {
    /// Set up storage for path elements
    pub fn new() -> Self {
        Self(SlotMap::new())
    }
}

/// Link to a PathElem in PathElemStorage
///
/// Behaves like an Rc, but uses PathElemStorage as a backing store instead of
/// the system memory allocator.
///
pub struct PathLink {
    /// Key of the target PathElem in the underlying PathElemStorage
    key: DefaultKey,

    /// In debug mode, we make sure that PathElems are correctly disposed of
    #[cfg(debug_assertions)]
    disposed: bool,
}
//
impl PathLink {
    /// Record a new path element
    pub fn new(
        storage: &mut PathElemStorage,
        curr_step: FeedPair,
        curr_cost: cache::Cost,
        prev_steps: Option<PathLink>,
    ) -> Self {
        let key = storage.0.insert(PathElem {
            reference_count: 1,
            curr_step,
            curr_cost,
            prev_steps,
        });
        Self {
            key,
            #[cfg(debug_assertions)]
            disposed: false,
        }
    }

    /// Read-only access to a path element from storage
    pub fn get<'storage>(&self, storage: &'storage PathElemStorage) -> &'storage PathElem {
        self.debug_assert_valid();
        &storage.0[self.key]
    }

    /// Mutable access to a path element from storage
    pub fn get_mut<'storage>(
        &self,
        storage: &'storage mut PathElemStorage,
    ) -> &'storage mut PathElem {
        self.debug_assert_valid();
        &mut storage.0[self.key]
    }

    /// Make a new PathLink pointing to the same PathElem
    pub fn clone(&self, storage: &mut PathElemStorage) -> Self {
        self.debug_assert_valid();
        self.get_mut(storage).reference_count += 1;
        Self {
            key: self.key,
            #[cfg(debug_assertions)]
            disposed: false,
        }
    }

    /// Invalidate a PathLink, possibly disposing of the underlying storage
    pub fn dispose(&mut self, storage: &mut PathElemStorage) {
        // Manual tail call optimization of recursive PathLink::dispose()
        self.debug_assert_valid();
        let mut disposed_key = Some(self.key);
        self.debug_invalidate();
        while let Some(key) = disposed_key.take() {
            // Reduce refcount of current path element
            let path_elem = &mut storage.0[key];
            path_elem.reference_count -= 1;

            // If no one uses that path element anymore...
            if path_elem.reference_count == 0 {
                // ...prepare to recursively dispose of any previous elements...
                if let Some(prev_link) = path_elem.prev_steps.as_mut() {
                    prev_link.debug_assert_valid();
                    disposed_key = Some(prev_link.key);
                    prev_link.debug_invalidate();
                }

                // ...and discard current path element
                storage.0.remove(key);
            }
        }
    }

    /// In debug mode, make sure that this path link is still valid
    #[inline(always)]
    fn debug_assert_valid(&self) {
        if cfg!(debug_assertions) {
            debug_assert!(!self.disposed);
        }
    }

    /// In debug mode, invalidate this path link
    #[inline(always)]
    fn debug_invalidate(&mut self) {
        if cfg!(debug_assertions) {
            self.disposed = true;
        }
    }
}
//
#[cfg(debug_assertions)]
impl Drop for PathLink {
    fn drop(&mut self) {
        assert!(
            self.disposed,
            "PathLink dropped without having been properly disposed of"
        );
    }
}

/// Reference-counted PartialPath path element
pub struct PathElem {
    /// Number of references to that path element in existence
    ///
    /// This is 1 when a path is created, increases to N when a path is forked
    /// into sub-paths, and once it drops to 0, all sub-paths have been fully
    /// explored, and this path an all of its parent paths can be disposed of.
    ///
    reference_count: u8,

    /// Last step that was taken on that path
    pub curr_step: FeedPair,

    /// Total cache cost after taking this step
    pub curr_cost: cache::Cost,

    /// Previous steps that were taken on this path
    pub prev_steps: Option<PathLink>,
}
//
#[cfg(debug_assertions)]
impl Drop for PathElem {
    fn drop(&mut self) {
        assert_eq!(
            self.reference_count, 0,
            "PathElem dropped while still referenced (according to refcount)"
        );
    }
}
