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
/// You must call dispose() before dropping this struct, otherwise you'll have a
/// memory leak on your hands. This is checked in debug build.
///
pub(super) struct PathLink {
    /// Key of the target PathElem in the underlying PathElemStorage
    key: DefaultKey,

    /// In debug mode, we make sure that PathElems are correctly disposed of
    #[cfg(debug_assertions)]
    disposed: DisposedStatus,
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
            disposed: DisposedStatus::Live,
        }
    }

    /// Read-only access to a path element from storage
    pub fn get<'storage>(&self, storage: &'storage PathElemStorage) -> &'storage PathElem {
        self.debug_assert_valid();
        // This is safe because...
        // - The PathElemStorage newtype does not let the user manipulate the
        //   storage, so PathLink is the only user-accessible way to insert and
        //   remove PathElems from storage.
        // - The reference counting protocol ensures that as long as there is
        //   a live PathLink to a storage location (weak clones aside), the
        //   corresponding PathElem cannot be destroyed.
        debug_assert!(storage.0.contains_key(self.key));
        unsafe { storage.0.get_unchecked(self.key) }
    }

    /// Mutable access to a path element from storage
    pub fn get_mut<'storage>(
        &self,
        storage: &'storage mut PathElemStorage,
    ) -> &'storage mut PathElem {
        self.debug_assert_valid();
        Self::get_mut_impl(self.key, storage)
    }

    /// Implementation of get_mut()
    fn get_mut_impl<'storage>(
        key: DefaultKey,
        storage: &'storage mut PathElemStorage,
    ) -> &'storage mut PathElem {
        // This is safe for the same reasons that get() is safe
        debug_assert!(storage.0.contains_key(key));
        unsafe { storage.0.get_unchecked_mut(key) }
    }

    /// Instruct the CPU to prefetch the data behind this PathLink, if supported
    pub fn prefetch(&self, storage: &PathElemStorage) {
        self.debug_assert_valid();
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
            // This is safe because we only execute the code on x86_64
            unsafe { _mm_prefetch(self.get(storage) as *const _ as *const i8, _MM_HINT_T0) };
        }
    }

    /// Make a new PathLink pointing to the same PathElem
    pub fn clone(&self, storage: &mut PathElemStorage) -> Self {
        self.debug_assert_valid();
        self.get_mut(storage).reference_count += 1;
        Self {
            key: self.key,
            #[cfg(debug_assertions)]
            disposed: DisposedStatus::Live,
        }
    }

    /// Make a new PathLink pointing to the same PathElem without incrementing
    /// the underlying reference count
    ///
    /// Using this PathLink after all other "strong" PathLinks have been
    /// destroyed will trigger undefined behavior, so creating it obviously is
    /// an unsafe operation.
    ///
    /// The output PathLink must not be disposed of.
    ///
    pub unsafe fn weak_clone(&self) -> Self {
        self.debug_assert_valid();
        Self {
            key: self.key,
            #[cfg(debug_assertions)]
            disposed: DisposedStatus::Weak,
        }
    }

    /// Invalidate a PathLink, possibly disposing of the underlying storage
    pub fn dispose(&mut self, storage: &mut PathElemStorage) {
        // Manual tail call optimization of recursive PathLink::dispose()
        self.debug_assert_live();
        let mut disposed_key = Some(self.key);
        self.debug_invalidate();
        while let Some(key) = disposed_key.take() {
            // Reduce refcount of current path element
            let path_elem = Self::get_mut_impl(key, storage);
            path_elem.reference_count -= 1;

            // If no one uses that path element anymore...
            if path_elem.reference_count == 0 {
                // ...prepare to recursively dispose of any previous elements...
                if let Some(prev_link) = path_elem.prev_steps.as_mut() {
                    prev_link.debug_assert_live();
                    disposed_key = Some(prev_link.key);
                    prev_link.debug_invalidate();
                }

                // ...and discard current path element
                storage.0.remove(key);
            }
        }
    }

    /// In debug mode, make sure that this path link can be dereferenced
    #[inline(always)]
    fn debug_assert_valid(&self) {
        #[cfg(debug_assertions)]
        {
            debug_assert_ne!(self.disposed, DisposedStatus::Dead);
        }
    }

    /// In debug mode, make sure that this path link should be disposed of
    #[inline(always)]
    fn debug_assert_live(&self) {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(self.disposed, DisposedStatus::Live);
        }
    }

    /// In debug mode, invalidate this path link
    #[inline(always)]
    fn debug_invalidate(&mut self) {
        #[cfg(debug_assertions)]
        {
            self.disposed = DisposedStatus::Dead;
        }
    }
}
//
#[cfg(debug_assertions)]
impl Drop for PathLink {
    fn drop(&mut self) {
        assert_ne!(
            self.disposed,
            DisposedStatus::Live,
            "PathLink dropped without having been properly disposed of"
        );
    }
}

#[cfg(debug_assertions)]
/// Value of PathLink::disposed
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum DisposedStatus {
    /// PathLink points to live data and that data must be disposed of.
    Live,

    /// PathLink was created using weak_clone(), we can assert that it points
    /// to valid data but we should not liberate said data.
    Weak,

    /// PathLink has been disposed of and should not be used anymore.
    Dead,
}

/// Reference-counted PartialPath path element
pub(super) struct PathElem {
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
