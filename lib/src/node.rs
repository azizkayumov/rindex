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
    pub fn new(
        slot_id: usize,
        height: usize,
        parent: usize,
        sphere: Sphere<D>,
        children: Vec<usize>,
    ) -> Node<D> {
        Node {
            slot_id,
            height,
            parent,
            sphere,
            children,
        }
    }

    #[must_use]
    pub fn point(point: [f64; D]) -> Node<D> {
        Self::new(usize::MAX, 0, usize::MAX, Sphere::point(point), Vec::new())
    }

    #[must_use]
    pub fn leaf() -> Node<D> {
        Self::new(usize::MAX, 1, usize::MAX, Sphere::default(), Vec::new())
    }

    #[must_use]
    pub fn internal(children: Vec<usize>) -> Node<D> {
        Self::new(usize::MAX, 0, usize::MAX, Sphere::default(), children)
    }

    #[must_use]
    pub fn is_point(&self) -> bool {
        self.height == 0
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
