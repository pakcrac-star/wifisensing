//! ============================================================================
//! Module 2: Circular Buffer Theory & Temporal Memory (`ringbuffers.rs`)
//! ============================================================================
//!
//! Purpose
//! -------
//! Provides a fixed-size circular buffer (ring buffer) for storing temporal
//! histories without dynamic memory allocation.
//!
//! Mathematical Model
//! ------------------
//! For a history length N, the buffer represents the discrete-time sequence
//!
//!     x[n-N+1], ..., x[n-1], x[n]
//!
//! where x[n] is the newest observation.
//!
//! Design Guarantees
//! -----------------
//! • O(1) insertion.
//! • O(1) delayed lookup.
//! • Zero heap allocation.
//! • Fully compatible with #![no_std].
//!
//! Invariants
//! ----------
//! • Stores data only.
//! • Performs no DSP, filtering, or physical interpretation.
//! ============================================================================



/// Fixed-size circular buffer.
///
/// `T` is the stored data type.
/// `N` is the maximum history length.
#[derive(Debug, Clone)]
pub struct RingBuffer<T, const N: usize> {
    /// Fixed storage.
    buffer: [T; N],
    /// Index where the next value will be written.
    head: usize,
    /// Number of valid elements currently stored.
    count: usize,
}

impl<T: Copy + Default, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, const N: usize> RingBuffer<T, N> {
    /// Creates an empty ring buffer.
    pub fn new() -> Self {
        // Prevent instantiation of a 0-length buffer at compile/init time.
        assert!(N > 0, "RingBuffer capacity 'N' must be strictly greater than 0");
        
        Self {
            buffer: [T::default(); N],
            head: 0,
            count: 0,
        }
    }

    /// Inserts the newest sample x[n].
    ///
    /// Complexity: O(1)
    #[inline]
    pub fn push(&mut self, value: T) {
        self.buffer[self.head] = value;

        self.head += 1;
        
        // Faster than `self.head = (self.head + 1) % N` on embedded targets
        if self.head == N {
            self.head = 0;
        }

        if self.count < N {
            self.count += 1;
        }
    }

    /// Returns x[n-delay].
    ///
    /// delay = 0 returns the newest sample.
    ///
    /// Returns `None` if the requested delay exceeds the currently stored history.
    /// Complexity: O(1)
    #[inline]
    pub fn get_delayed(&self, delay: usize) -> Option<T> {
        if delay >= self.count {
            return None;
        }

        // Avoid expensive Modulo `% N` math. 
        // Since `head` points to the *next* insertion index, we step backward.
        let index = if self.head > delay {
            self.head - 1 - delay
        } else {
            N + self.head - 1 - delay
        };

        Some(self.buffer[index])
    }

    /// Returns the newest sample.
    #[inline]
    pub fn latest(&self) -> Option<T> {
        self.get_delayed(0)
    }

    /// Returns the number of valid samples.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns the maximum capacity.
    #[inline]
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns true when no valid samples are stored.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns true once the history reaches maximum capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count == N
    }

    /// Removes all stored history.
    ///
    /// Underlying memory is intentionally left unchanged because `count`
    /// defines the valid history region.
    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }

    /// Alias for `clear()`.
    #[inline]
    pub fn reset(&mut self) {
        self.clear();
    }
}
