use super::sphere::Sphere;

pub struct InsertionEntry<const D: usize> {
    pub idx: usize,
    pub parent_height: usize,
    pub sphere: Sphere<D>,
}

pub struct Node<const D: usize> {
    pub idx: usize,
    pub parent: usize,
    pub height: usize,
    pub sphere: Sphere<D>,
    pub variance: [f64; D],
    pub children: Vec<usize>,
    pub bound: f64,
}

impl<const D: usize> Node<D> {
    pub fn new(idx: usize, parent: usize, height: usize, sphere: Sphere<D>) -> Node<D> {
        Node {
            idx,
            parent,
            height,
            sphere,
            variance: [f64::INFINITY; D],
            children: Vec::new(),
            bound: f64::INFINITY,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.height == 0
    }
}
