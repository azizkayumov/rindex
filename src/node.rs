use crate::distance::euclidean;

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

#[derive(Clone, Copy)]
pub struct Sphere<const D: usize> {
    pub center: [f64; D],
    pub radius: f64,
}

impl<const D: usize> Sphere<D> {
    pub fn new(center: [f64; D], radius: f64) -> Sphere<D> {
        Sphere { center, radius }
    }

    pub fn min_distance(&self, other: &[f64; D]) -> f64 {
        (euclidean(&self.center, other) - self.radius).max(0.)
    }
}
