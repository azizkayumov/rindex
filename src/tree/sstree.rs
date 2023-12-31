use crate::{distance::euclidean, Index};
use conv::ValueFrom;
use ordered_float::{Float, OrderedFloat};
use std::collections::BinaryHeap;

use super::{
    node::{InsertionEntry, Node},
    sphere::Sphere,
};

pub struct SSTree<const D: usize> {
    k: usize,
    data: Vec<[f64; D]>,
    neighbors: Vec<BinaryHeap<(OrderedFloat<f64>, usize)>>,
    root: usize,
    node_max_entries: usize,
    nodes: Vec<Node<D>>,
}

impl<const D: usize> Index<D> for SSTree<D> {
    fn insert(&mut self, point: [f64; D]) {
        // Insert the new point
        let new_point_index = self.data.len();
        self.data.push(point);

        // Compute the core distance of the new point
        self.neighbors
            .push(BinaryHeap::from(vec![(OrderedFloat(0.), new_point_index)]));
        let new_point_neighbors = self.query(&point);
        for (neighbor_idx, _) in new_point_neighbors {
            self.add_core(new_point_index, neighbor_idx);
        }

        let mut reinsert_entries = vec![InsertionEntry {
            idx: new_point_index,
            parent_height: 0,
            sphere: Sphere::new(point, 0.0),
        }];
        let mut reinsert_height = 0;
        while let Some(entry) = reinsert_entries.pop() {
            let new_reinsert_entries = self.insert_recursive(entry, self.root, reinsert_height);
            reinsert_entries.extend(new_reinsert_entries);
            reinsert_height += 1;
        }

        if self.nodes[self.root].children.len() > self.node_max_entries {
            let old_root_idx = self.root;
            let sibling_entry = self.split(self.root);
            let new_root_idx = self.nodes.len();
            let mut new_root = Node::new(
                new_root_idx,
                usize::MAX,
                self.nodes[old_root_idx].height + 1,
                Sphere::new(self.nodes[sibling_entry.idx].sphere.center, 0.0),
            );
            new_root.children = vec![old_root_idx, sibling_entry.idx];
            self.nodes[old_root_idx].parent = new_root_idx;
            self.nodes[sibling_entry.idx].parent = new_root_idx;
            self.nodes.push(new_root);
            self.reshape(new_root_idx);
            self.root = new_root_idx;
        }
        self.update_core(self.root, new_point_index);
    }

    fn rknn(&self, point: [f64; D]) -> Vec<usize> {
        let mut result = Vec::new();
        self.rknn_recursive(self.root, &point, &mut result);
        result
    }

    fn query_range(&self, point_index: usize, range: f64) -> Vec<usize> {
        let mut neighbors = Vec::new();
        self.query_range_recursive(self.root, point_index, range, &mut neighbors);
        neighbors
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

    fn num_points(&self) -> usize {
        self.data.len()
    }
}

impl<const D: usize> SSTree<D> {
    #[must_use]
    pub fn new(k: usize) -> Self {
        let root = Node::new(0, usize::MAX, 0, Sphere::new([f64::INFINITY; D], 0.0));
        Self {
            k,
            data: Vec::new(),
            neighbors: Vec::new(),
            root: 0,
            node_max_entries: 2 * k + 1,
            nodes: vec![root],
        }
    }

    fn rknn_recursive(&self, node_idx: usize, point: &[f64; D], rneighbors: &mut Vec<usize>) {
        let distance_to_node = self.nodes[node_idx].sphere.min_distance(point);
        if distance_to_node > self.nodes[node_idx].bound {
            return;
        }

        if self.nodes[node_idx].is_leaf() {
            for neighbor_idx in &self.nodes[node_idx].children {
                let distance = euclidean(point, &self.data[*neighbor_idx]);
                if distance < self.core_distance_of(*neighbor_idx) {
                    rneighbors.push(*neighbor_idx);
                }
            }
        } else {
            for child_idx in &self.nodes[node_idx].children {
                self.rknn_recursive(*child_idx, point, rneighbors);
            }
        }
    }

    fn query_range_recursive(
        &self,
        node_idx: usize,
        point_index: usize,
        range: f64,
        neighbors: &mut Vec<usize>,
    ) {
        let distance_to_node = self.nodes[node_idx]
            .sphere
            .min_distance(&self.data[point_index]);
        if distance_to_node > range {
            return;
        }

        if self.nodes[node_idx].is_leaf() {
            for neighbor_idx in &self.nodes[node_idx].children {
                let distance_to_neighbor =
                    euclidean(&self.data[point_index], &self.data[*neighbor_idx]);
                if distance_to_neighbor <= range {
                    neighbors.push(*neighbor_idx);
                }
            }
        } else {
            for child_idx in &self.nodes[node_idx].children {
                self.query_range_recursive(*child_idx, point_index, range, neighbors);
            }
        }
    }

    fn query(&mut self, point: &[f64; D]) -> Vec<(usize, f64)> {
        if self.root == usize::MAX {
            return Vec::new();
        }
        let mut neighbors = BinaryHeap::new();
        self.query_recursive(point, self.root, &mut neighbors);
        neighbors
            .into_sorted_vec()
            .into_iter()
            .map(|(distance, idx)| (idx, distance.into_inner()))
            .collect::<Vec<(usize, f64)>>()
    }

    fn query_recursive(
        &mut self,
        point: &[f64; D],
        node_idx: usize,
        neighbors: &mut BinaryHeap<(OrderedFloat<f64>, usize)>,
    ) {
        let distance_to_node = self.nodes[node_idx].sphere.min_distance(point);
        let mut kth_distance = f64::INFINITY;
        if neighbors.len() == self.k {
            kth_distance = *neighbors.peek().unwrap().0;
        }
        if distance_to_node >= kth_distance {
            return;
        }

        if self.nodes[node_idx].is_leaf() {
            for point_idx in &self.nodes[node_idx].children {
                let distance_to_neighbor = euclidean(point, &self.data[*point_idx]);
                if neighbors.len() == self.k {
                    kth_distance = *neighbors.peek().unwrap().0;
                }
                if neighbors.len() < self.k {
                    neighbors.push((OrderedFloat(distance_to_neighbor), *point_idx));
                } else if distance_to_neighbor < kth_distance {
                    neighbors.pop();
                    neighbors.push((OrderedFloat(distance_to_neighbor), *point_idx));
                }
            }
        } else {
            let mut to_visit = Vec::new();
            for child_idx in &self.nodes[node_idx].children {
                let distance_to_child = self.nodes[*child_idx].sphere.min_distance(point);
                if distance_to_child < kth_distance {
                    to_visit.push((OrderedFloat(distance_to_child), *child_idx));
                }
            }
            to_visit.sort();

            for (child_distance, child_index) in to_visit {
                if neighbors.len() == self.k {
                    kth_distance = *neighbors.peek().unwrap().0;
                }
                if child_distance.0 >= kth_distance {
                    break;
                }
                self.query_recursive(point, child_index, neighbors);
            }
        }
    }

    fn update_bound(&mut self, node_idx: usize) {
        let mut bound: f64 = 0.0;
        for child_idx in &self.nodes[node_idx].children {
            let child_bound = if self.nodes[node_idx].is_leaf() {
                self.core_distance_of(*child_idx)
            } else {
                self.nodes[*child_idx].bound
            };
            bound = bound.max(child_bound);
        }
        self.nodes[node_idx].bound = bound;
    }

    fn add_core(&mut self, point_index: usize, neighbor_index: usize) -> bool {
        let cur_core_distance = self.core_distance_of(point_index);
        let distance = OrderedFloat(euclidean(
            &self.data[point_index],
            &self.data[neighbor_index],
        ));
        if distance.0 >= cur_core_distance {
            return false;
        }
        self.neighbors[point_index].push((distance, neighbor_index));
        if self.neighbors[point_index].len() > self.k {
            self.neighbors[point_index].pop();
        }
        true
    }

    fn update_core(&mut self, node_idx: usize, point_index: usize) -> bool {
        let distance_to_node = self.nodes[node_idx]
            .sphere
            .min_distance(&self.data[point_index]);
        if distance_to_node > self.nodes[node_idx].bound {
            return false;
        }

        let mut updated = false;
        let to_visit = self.nodes[node_idx].children.clone();
        if self.nodes[node_idx].is_leaf() {
            for neighbor_idx in to_visit {
                if neighbor_idx != point_index {
                    let cur = self.add_core(neighbor_idx, point_index);
                    updated = updated || cur;
                }
            }
        } else {
            for child_idx in to_visit {
                let cur = self.update_core(child_idx, point_index);
                updated = updated || cur;
            }
        }
        if updated {
            self.update_bound(node_idx);
        }

        updated
    }

    fn insert_recursive(
        &mut self,
        entry: InsertionEntry<D>,
        node_idx: usize,
        reinsert_height: usize,
    ) -> Vec<InsertionEntry<D>> {
        assert!(
            self.nodes[node_idx].height >= entry.parent_height,
            "Node height is lower than the insertion height"
        );
        if self.nodes[node_idx].height == entry.parent_height {
            self.nodes[node_idx].children.push(entry.idx);
            self.reshape(node_idx);
            if self.nodes[node_idx].children.len() <= self.node_max_entries || node_idx == self.root
            {
                return Vec::new();
            }

            if self.nodes[node_idx].height == reinsert_height {
                // reinsert
                self.pop_farthest_children(node_idx, self.k)
            } else {
                // split
                vec![self.split(node_idx)]
            }
        } else {
            let closest_child_idx = self.choose_subtree(node_idx, &entry.sphere.center);
            let reinsert_entries = self.insert_recursive(entry, closest_child_idx, reinsert_height);
            self.reshape(node_idx);
            reinsert_entries
        }
    }

    fn choose_subtree(&self, node_idx: usize, point: &[f64; D]) -> usize {
        let mut closest_child_idx = usize::MAX;
        let mut closest_child_distance = OrderedFloat::max_value();
        let mut closest_center_distance = OrderedFloat::max_value();
        for child_idx in &self.nodes[node_idx].children {
            let distance = OrderedFloat(self.nodes[*child_idx].sphere.min_distance(point));
            let center_distance =
                OrderedFloat(euclidean(&self.nodes[*child_idx].sphere.center, point));
            if distance < closest_child_distance
                || distance == closest_child_distance && center_distance < closest_center_distance
            {
                closest_child_distance = distance;
                closest_child_idx = *child_idx;
                closest_center_distance = center_distance;
            }
        }
        closest_child_idx
    }

    fn pop_farthest_children(&mut self, node_idx: usize, count: usize) -> Vec<InsertionEntry<D>> {
        assert!(2 * count <= self.nodes[node_idx].children.len());
        let parent_centroid = self.nodes[node_idx].sphere.center;
        let mut children = self.nodes[node_idx].children.clone();
        children.sort_by_key(|child_idx| {
            OrderedFloat(
                self.child_sphere(node_idx, *child_idx)
                    .min_distance(&parent_centroid),
            )
        });
        let mut far_children = Vec::new();
        while far_children.len() < count {
            far_children.push(children.pop().unwrap());
        }
        self.nodes[node_idx].children = children;
        self.reshape(node_idx);

        let mut reinsert_entries = Vec::new();
        for far_child in far_children {
            let sphere = if self.nodes[node_idx].is_leaf() {
                Sphere::new(self.data[far_child], 0.0)
            } else {
                self.nodes[far_child].sphere
            };
            let entry = InsertionEntry {
                idx: far_child,
                sphere,
                parent_height: self.nodes[node_idx].height,
            };
            reinsert_entries.push(entry);
        }

        reinsert_entries
    }

    fn split(&mut self, node_idx: usize) -> InsertionEntry<D> {
        let parent = self.nodes[node_idx].parent;

        // choose split axis
        let split_axis = self.choose_split_axis(node_idx);

        // sort the children by the split axis
        let mut children = self.nodes[node_idx].children.clone();
        children.sort_by_key(|child_idx| {
            OrderedFloat(self.child_centroid(node_idx, *child_idx)[split_axis])
        });
        self.nodes[node_idx].children = children;

        // choose split index
        let mut split_index = self.choose_split_index(node_idx);

        let left_centroid = self.calculate_center(node_idx, 0, split_index);
        let right_centroid =
            self.calculate_center(node_idx, split_index, self.nodes[node_idx].children.len());
        let left_distance = euclidean(&left_centroid, &self.nodes[node_idx].sphere.center);
        let right_distance = euclidean(&right_centroid, &self.nodes[node_idx].sphere.center);
        if left_distance > right_distance {
            self.nodes[node_idx].children.reverse();
            split_index = self.nodes[node_idx].children.len() - split_index;
        }

        // create the sibling node
        let mut sibling_children = Vec::new();
        while self.nodes[node_idx].children.len() > split_index {
            sibling_children.push(self.nodes[node_idx].children.pop().unwrap());
        }
        self.reshape(node_idx);

        let sibling_idx = self.nodes.len();
        let sibling_sphere = Sphere::new(left_centroid, 0.);
        let mut sibling = Node::new(
            sibling_idx,
            parent,
            self.nodes[node_idx].height,
            sibling_sphere,
        );
        sibling.children = sibling_children;
        self.nodes.push(sibling);
        self.reshape(sibling_idx);

        let sibling_sphere = self.nodes[sibling_idx].sphere;
        InsertionEntry {
            idx: sibling_idx,
            sphere: sibling_sphere,
            parent_height: self.nodes[node_idx].height + 1,
        }
    }

    fn choose_split_index(&self, node_idx: usize) -> usize {
        assert!(self.nodes[node_idx].children.len() >= self.node_max_entries);
        let num_children = self.nodes[node_idx].children.len();
        let mut selected_index = num_children / 2;
        let mut min_variance = f64::INFINITY;

        let start = self.k;
        let end = num_children - self.k;
        for index in start..end {
            let left_variance = self.calculate_variance(node_idx, 0, index);
            let right_variance = self.calculate_variance(node_idx, index, num_children);
            let mut cur_variance = 0.0;
            for d in 0..D {
                cur_variance += left_variance[d] + right_variance[d];
            }
            if cur_variance < min_variance {
                min_variance = cur_variance;
                selected_index = index;
            }
        }
        selected_index
    }

    fn choose_split_axis(&self, node_idx: usize) -> usize {
        assert!(self.nodes[node_idx].children.len() >= self.node_max_entries);
        let variance = self.calculate_variance(node_idx, 0, self.nodes[node_idx].children.len());
        variance
            .iter()
            .enumerate()
            .max_by_key(|(_, variance)| OrderedFloat(**variance))
            .unwrap()
            .0
    }

    fn calculate_center(&self, node_idx: usize, from: usize, to: usize) -> [f64; D] {
        let mut center = [0.0; D];
        let mut weight = 0;
        for i in from..to {
            let child_idx = self.nodes[node_idx].children[i];
            let child_weight = self.child_weight(node_idx, child_idx);
            let child_center = self.child_centroid(node_idx, child_idx);
            weight += child_weight;
            for dim in 0..D {
                center[dim] += child_center[dim] * f64::value_from(child_weight).unwrap();
            }
        }
        for dim in center.iter_mut().take(D) {
            *dim /= f64::value_from(weight).unwrap();
        }
        center
    }

    fn calculate_variance(&self, node_idx: usize, from: usize, to: usize) -> [f64; D] {
        let mean = self.calculate_center(node_idx, from, to);
        let mut variance = [0.0; D];
        let mut num_entries = 0;
        for i in from..to {
            let child_idx = self.nodes[node_idx].children[i];
            let child_centroid = self.child_centroid(node_idx, child_idx);
            let child_num_entries = self.child_weight(node_idx, child_idx);
            for axis in 0..D {
                variance[axis] += (child_centroid[axis] - mean[axis]).powi(2)
                    * f64::value_from(child_num_entries).unwrap();
                if !self.nodes[node_idx].is_leaf() {
                    variance[axis] += self.nodes[child_idx].variance[axis]
                        * f64::value_from(child_num_entries).unwrap();
                }
            }
            num_entries += child_num_entries;
        }
        for var in variance.iter_mut().take(D) {
            *var /= f64::value_from(num_entries).unwrap();
        }
        variance
    }

    fn child_sphere(&self, node_idx: usize, child_idx: usize) -> Sphere<D> {
        if self.nodes[node_idx].is_leaf() {
            Sphere::new(self.data[child_idx], 0.0)
        } else {
            self.nodes[child_idx].sphere
        }
    }

    fn child_centroid(&self, node_idx: usize, child_idx: usize) -> [f64; D] {
        if self.nodes[node_idx].is_leaf() {
            self.data[child_idx]
        } else {
            self.nodes[child_idx].sphere.center
        }
    }

    fn child_radius(&self, node_idx: usize, child_idx: usize) -> f64 {
        if self.nodes[node_idx].is_leaf() {
            0.0
        } else {
            self.nodes[child_idx].sphere.radius
        }
    }

    fn child_weight(&self, node_idx: usize, child_idx: usize) -> usize {
        if self.nodes[node_idx].is_leaf() {
            1
        } else {
            self.nodes[child_idx].children.len()
        }
    }

    fn reshape(&mut self, node_idx: usize) {
        let center = self.calculate_center(node_idx, 0, self.nodes[node_idx].children.len());
        let mut radius: f64 = 0.;
        let children = self.nodes[node_idx].children.clone();

        for child_idx in children {
            let child_centroid = self.child_centroid(node_idx, child_idx);
            let child_radius = self.child_radius(node_idx, child_idx);
            radius = radius.max(euclidean(&center, &child_centroid) + child_radius);
            if !self.nodes[node_idx].is_leaf() {
                self.nodes[child_idx].parent = node_idx;
            }
        }
        self.nodes[node_idx].sphere = Sphere::new(center, radius);
        self.nodes[node_idx].variance =
            self.calculate_variance(node_idx, 0, self.nodes[node_idx].children.len());
        self.update_bound(node_idx);
    }
}
