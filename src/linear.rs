use crate::{distance::euclidean, index::Index};
use ordered_float::OrderedFloat;
use std::collections::BinaryHeap;

pub struct LinearIndex<const D: usize> {
    k: usize,
    data: Vec<[f64; D]>,
    neighbors: Vec<BinaryHeap<(OrderedFloat<f64>, usize)>>,
}

impl<const D: usize> LinearIndex<D> {
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            data: Vec::new(),
            neighbors: Vec::new(),
        }
    }

    fn add_neighbor(&mut self, a: usize, b: usize) -> bool {
        let distance = euclidean(&self.data[a], &self.data[b]);
        let core_a = self.core_distance_of(a);
        let core_b = self.core_distance_of(b);
        if a != b && distance < core_b {
            self.neighbors[b].push((OrderedFloat(distance), a));
            if self.neighbors[b].len() > self.k {
                self.neighbors[b].pop();
            }
        }

        if distance < core_a {
            self.neighbors[a].push((OrderedFloat(distance), b));
            if self.neighbors[a].len() > self.k {
                self.neighbors[a].pop();
            }
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn num_points(&self) -> usize {
        self.data.len()
    }
}

impl<const D: usize> Index<D> for LinearIndex<D> {
    fn insert(&mut self, point: [f64; D]) -> Vec<usize> {
        // Append the point to the data.
        let new_point_index = self.data.len();
        self.data.push(point);
        self.neighbors.push(BinaryHeap::new());

        // Find the reverse k-nearest neighbors of the point and update its core distance.
        let mut rknns = Vec::new();
        for neighbor_idx in 0..self.num_points() {
            if self.add_neighbor(neighbor_idx, new_point_index) {
                rknns.push(neighbor_idx);
            }
        }
        rknns
    }

    fn query_range(&self, point_index: usize, range: f64) -> Vec<usize> {
        let mut result = Vec::new();
        for i in 0..self.num_points() {
            if euclidean(&self.data[point_index], &self.data[i]) <= range {
                result.push(i);
            }
        }
        result
    }

    fn core_distance_of(&self, point_index: usize) -> f64 {
        if self.neighbors[point_index].len() != self.k {
            return f64::INFINITY;
        }
        self.neighbors[point_index].peek().unwrap().0.into_inner()
    }

    fn neighbors_of(&self, point_index: usize) -> Vec<usize> {
        self.neighbors[point_index]
            .iter()
            .map(|(_, neighbor_index)| *neighbor_index)
            .collect()
    }
}

#[cfg(test)]
pub mod tests {
    use crate::Index;

    #[test]
    pub fn test_linear() {
        let mut index = super::LinearIndex::new(3);
        let rknn = index.insert([0.0, 0.0]);
        assert_eq!(rknn.len(), 1);
        assert!(rknn.contains(&0));

        for i in 1..100 {
            let point = [0., i as f64];

            // Since points are inserted in order, the reverse k-nearest neighbors should
            // be the previously inserted point and the new point.
            let rknns = index.insert(point);
            assert!(rknns.contains(&(i - 1)));
            assert!(rknns.contains(&i));
        }
    }
}
