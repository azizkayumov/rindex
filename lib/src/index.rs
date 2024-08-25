pub struct Index {
    num_slots: usize,
    deleted_slots: Vec<usize>,
}

impl Index {
    #[must_use]
    pub fn new() -> Index {
        Index {
            num_slots: 0,
            deleted_slots: Vec::new(),
        }
    }

    // Allocate or reuse a slot of the index.
    pub fn insert(&mut self) -> usize {
        if self.deleted_slots.is_empty() {
            let slot_id = self.num_slots;
            self.num_slots += 1;
            slot_id
        } else {
            self.deleted_slots.pop().unwrap()
        }
    }

    // Delete a slot from the index.
    pub fn delete(&mut self, slot_id: usize) {
        self.deleted_slots.push(slot_id);
    }
}

impl Default for Index {
    fn default() -> Self {
        Self::new()
    }
}
