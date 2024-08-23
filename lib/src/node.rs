use std::usize;

use crate::sphere::Sphere;

pub struct Node<const D: usize> {
    pub slot_id: usize,
    pub height: usize,
    pub parent: usize,
    pub sphere: Sphere<D>,
    pub children: Vec<usize>,
}

impl<const D: usize> Node<D> {
    #[must_use]
    pub fn new(slot_id: usize, height: usize, parent: usize, sphere: Sphere<D>) -> Node<D> {
        Node {
            slot_id,
            height,
            parent,
            sphere,
            children: Vec::new(),
        }
    }

    #[must_use]
    pub fn point(point: [f64; D]) -> Node<D> {
        Self::new(usize::MAX, 0, usize::MAX, Sphere::point(point))
    }
}

impl<const D: usize> Default for Node<D> {
    fn default() -> Self {
        Node {
            slot_id: usize::MAX,
            height: 0,
            parent: usize::MAX,
            sphere: Sphere::default(),
            children: Vec::new(),
        }
    }
}
