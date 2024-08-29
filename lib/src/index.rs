use crate::node::Node;

pub struct Index<const D: usize> {
    deleted_slots: Vec<usize>,
    pub nodes: Vec<Node<D>>,
}

impl<const D: usize> Default for Index<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> Index<D> {
    #[must_use]
    pub fn new() -> Self {
        Index {
            deleted_slots: Vec::new(),
            nodes: Vec::new(),
        }
    }

    // Allocate or reuse a slot of the index.
    pub fn insert(&mut self, mut node: Node<D>) -> usize {
        if self.deleted_slots.is_empty() {
            let slot_id = self.nodes.len();
            node.slot_id = slot_id;
            self.nodes.push(node);
            slot_id
        } else {
            let slot_id = self.deleted_slots.pop().unwrap();
            node.slot_id = slot_id;
            self.nodes[slot_id] = node;
            slot_id
        }
    }

    // Delete a slot from the index.
    pub fn delete(&mut self, slot_id: usize) {
        self.deleted_slots.push(slot_id);
    }
}
