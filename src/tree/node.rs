use super::sphere::Sphere;

pub struct InsertionEntry<const D: usize> {
    pub idx: usize,
    pub sphere: Sphere<D>,
    pub parent_height: usize,
}

pub struct Node<const D: usize> {
    pub idx: usize,
    pub height: usize,
    pub sphere: Sphere<D>,
    pub parent: usize,
    pub children: Vec<usize>,
    pub variance: [f64; D],
    pub bound: f64,
}

impl<const D: usize> Node<D> {
    pub fn new(idx: usize, height: usize, sphere: Sphere<D>, parent: usize) -> Node<D> {
        Node {
            idx,
            height,
            sphere,
            parent,
            children: Vec::new(),
            variance: [f64::INFINITY; D],
            bound: f64::INFINITY,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.height == 0
    }
}
