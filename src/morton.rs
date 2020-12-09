//! Implementation of Morton code decoding

use crate::FeedIdx;

// Number of bits in an integer of a certain size
const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

// Generate a mask with an alternate bit pattern of type 00110011...0011
const fn striped_mask(stripe_length: usize) -> FeedIdx {
    // Compute the total length of the output mask in bits
    let num_mask_bits = num_bits::<FeedIdx>();
    // TODO: Once assert in const is allowed, sanity check input
    // assert!(stripe_length.is_power_of_two() && stripe_length < num_mask_bits / 2);

    // Build a right-aligned stripe of ones of the right length
    let mut stripe = 0b1;
    let mut current_stripe_length = 1;
    while stripe_length != current_stripe_length {
        stripe |= stripe << current_stripe_length;
        current_stripe_length *= 2;
    }

    // Build a mask that alternates between stripes of zeros and stripes of
    // ones of that length, in 00110011...0011 order.
    let mut mask = stripe;
    let mut current_mask_length = 2 * stripe_length;
    while current_mask_length != num_mask_bits {
        mask |= mask << current_mask_length;
        current_mask_length *= 2;
    }
    mask
}

/// Decode an 2-dimensional Morton code into its two inner indices
///
/// A Morton code combines two integers with bit patterns [ a1 a2 ... aN ] and
/// [ b1 b2 ... bN ] into the interleaved bit pattern [ b1 a1 b2 a2 ... bN aN ].
/// Iterating over 2D indices in Morton code order produces a Z-shaped fractal
/// space-filling curve that has good spatial locality properties.
pub const fn decode_2d(code: FeedIdx) -> [FeedIdx; 2] {
    // Align the low-order bits of the two input sub-codes:
    // [ XX a1 XX a2 XX a3 XX a4 ... aN-1   XX aN ]
    // [ XX b1 XX b2 XX b3 XX b4 ... bN-1   XX bN ]
    let mut sub_codes = [code, code >> 1];
    let mut sub_code_idx = 0;
    while sub_code_idx < 2 {
        // Initially, we get an index's bits interleaved with irrelevant junk:
        // [ XX a1 XX a2 XX a3 XX a4 ... XX aN-1 XX aN ]
        // Let's clean that up by zeroing out the junk:
        // [  0 a1  0 a2  0 a3  0 a4 ...  0 aN-1  0 aN ]
        let mut sub_code = sub_codes[sub_code_idx] & striped_mask(1);
        // We will de-interleave the index by recursively grouping the bits that
        // we're interested in in pairs, groups of 4, and so on.
        // Initially, bits are isolated, so we have groups of one.
        // We're done once we have grouped half of the input bits together,
        // since the other bits are zeroes.
        let mut group_size = 1;
        while group_size != num_bits::<FeedIdx>() / 2 {
            // Duplicate the current bit pattern into neighboring zeroes on the
            // right in order to group pairs of subcode bits together
            // Iteration 1: [  0 a1 a1 a2 a2 a3 a3 a4 ... aN-2 aN-1 aN-1 aN ]
            // Iteration 2: [  0  0 a1 a2 a1 a2 a3 a4 ... aN-3 aN-2 aN-1 aN ]
            sub_code |= sub_code >> group_size;
            // Only keep a single copy of each bit group, zeroing out the rest
            // Iteration 1: [  0 a1  0 a2  0 a3  0 a4 ...    0 aN-1    0 aN ]
            // Iteration 2: [  0  0 a1 a2  0  0 a3 a4 ...    0    0 aN-1 aN ]
            sub_code &= striped_mask(group_size);
            // Repeat until all bits have been grouped together
            group_size *= 2;
        }
        // Record the decoded index and move to the next one
        sub_codes[sub_code_idx] = sub_code;
        sub_code_idx += 1;
    }
    sub_codes
}
