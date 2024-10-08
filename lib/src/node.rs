use crate::sphere::Sphere;
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;

pub struct Node<const D: usize> {
    pub slot_id: usize,
    pub height: usize,
    pub parent: usize,
    pub sphere: Sphere<D>,
    pub children: Vec<usize>,
    pub neighbors: BinaryHeap<(OrderedFloat<f64>, usize)>,
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
            neighbors: BinaryHeap::new(),
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
        Self::new(
            usize::MAX,
            usize::MAX,
            usize::MAX,
            Sphere::default(),
            children,
        )
    }

    #[must_use]
    pub fn min_distance(&self, point: &[f64; D]) -> f64 {
        self.sphere.min_distance(point)
    }

    #[must_use]
    pub fn max_distance(&self, point: &[f64; D]) -> f64 {
        self.sphere.max_distance(point)
    }

    #[must_use]
    pub fn bound(&self) -> f64 {
        if self.is_point() {
            self.neighbors
                .peek()
                .unwrap_or(&(OrderedFloat(f64::INFINITY), 0))
                .0
                 .0
        } else {
            self.sphere.bound
        }
    }

    #[must_use]
    pub fn is_point(&self) -> bool {
        self.height == 0
    }

    #[must_use]
    pub fn is_leaf(&self) -> bool {
        self.height == 1
    }

    #[must_use]
    pub fn is_deleted(&self) -> bool {
        self.slot_id == usize::MAX
    }

    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        let children = self
            .children
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<String>>()
            .join("+");
        format!(
            "{},{},{},{},{},{},{},{}\n",
            self.slot_id,
            self.height,
            self.parent,
            children,
            self.sphere.radius,
            self.sphere.center[0],
            self.sphere.center[1],
            self.bound(),
        )
    }
}

impl<const D: usize> Default for Node<D> {
    fn default() -> Self {
        Node {
            slot_id: usize::MAX,
            height: usize::MAX,
            parent: usize::MAX,
            sphere: Sphere::default(),
            children: Vec::new(),
            neighbors: BinaryHeap::new(),
        }
    }
}
