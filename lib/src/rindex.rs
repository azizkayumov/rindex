use core::f64;
use std::{collections::BinaryHeap, vec};

use crate::{distance::euclidean, index::Index, node::Node, sphere::Sphere};
use ordered_float::OrderedFloat;

/// Rindex: dynamic spatial index for efficiently maintaining *k* nearest neighbors graph of multi-dimensional clustered datasets.
///
/// # Examples
///
/// ```
/// use rindex::Rindex;
///
/// let k = 3;
/// let mut rindex = Rindex::new(k);
///
/// // Insert some points
/// let a = rindex.insert([1.0, 1.0]);
/// let b = rindex.insert([2.0, 2.0]);
/// let c = rindex.insert([3.0, 3.0]);
/// let d = rindex.insert([20.0, 20.0]);
///
/// // Check k nearest neighbors of point a
/// let (neighbors, distances) = rindex.neighbors_of(a);
/// assert_eq!(neighbors, vec![a, b, c]);
///
/// // Remove point b
/// rindex.delete(b);
///
/// // Check k nearest neighbors of point a again
/// let (neighbors, distances) = rindex.neighbors_of(a);
/// assert_eq!(neighbors, vec![a, c, d]); // b is not in the result
/// ```
pub struct Rindex<const D: usize> {
    min_fanout: usize,
    max_fanout: usize,
    reinsert_fanout: usize,
    reinsert_height: usize,
    k: usize,
    root: usize,
    index: Index<D>,
    num_points: usize,
}

impl<const D: usize> Default for Rindex<D> {
    fn default() -> Self {
        Rindex::new(10)
    }
}

// Public methods
impl<const D: usize> Rindex<D> {
    #[must_use]
    pub fn new_with_params(max_fanout: usize, k: usize) -> Option<Self> {
        if max_fanout < 4 || k < 1 {
            return None;
        }
        Some(Rindex {
            min_fanout: max_fanout / 2,
            max_fanout,
            reinsert_fanout: max_fanout / 3,
            reinsert_height: 1,
            k,
            root: usize::MAX,
            index: Index::new(),
            num_points: 0,
        })
    }

    #[must_use]
    pub fn new(k: usize) -> Self {
        Rindex {
            min_fanout: 5,
            max_fanout: 10,
            reinsert_fanout: 3,
            reinsert_height: 1,
            k,
            root: usize::MAX,
            index: Index::new(),
            num_points: 0,
        }
    }

    /// # Examples
    /// ```
    /// use rindex::Rindex;
    ///
    /// let mut rindex = Rindex::default();
    /// let a = rindex.insert([1.0, 1.0]);
    /// assert_eq!(rindex.point_at(a), &[1.0, 1.0]);
    #[must_use]
    pub fn insert(&mut self, point: [f64; D]) -> usize {
        self.num_points += 1;

        // Create the root node if it doesn't exist
        if self.root == usize::MAX {
            let node = Node::leaf();
            self.root = self.add_slot(node);
        }

        // Create a point node (reuse a slot in the node vector if possible)
        let slot_id = self.add_slot(Node::point(point));

        // Insert the point node into the tree
        let mut reinsert_list = vec![slot_id];
        self.reinsert_nodes(&mut reinsert_list);

        slot_id
    }

    /// # Examples
    /// ```
    /// use rindex::Rindex;
    ///
    /// let mut rindex = Rindex::default();
    /// let a = rindex.insert([1.0, 1.0]);
    /// assert_eq!(rindex.num_points(), 1);
    /// rindex.delete(a);
    /// assert_eq!(rindex.num_points(), 0);
    pub fn delete(&mut self, point_id: usize) {
        self.num_points -= 1;

        let deleted_point = self.index.nodes[point_id].sphere.center;
        let mut reinsert_list = self.delete_entry(point_id);
        self.reinsert_nodes(&mut reinsert_list);

        // Update the reverse neighbors of the deleted point
        let (reverse_neighbors, _) = self.query_reverse(&deleted_point);
        for r in &reverse_neighbors {
            self.index.nodes[*r].neighbors =
                BinaryHeap::from(vec![(OrderedFloat(f64::INFINITY), usize::MAX); self.k]);
        }
        let sphere = self.calculate_sphere(&reverse_neighbors);
        self.post_delete(self.root, &reverse_neighbors, &sphere);
    }

    /// # Examples
    /// ```
    /// use rindex::Rindex;
    ///
    /// let mut rindex = Rindex::default();
    /// let a = rindex.insert([1.0, 1.0]);
    /// let b = rindex.insert([2.0, 2.0]);
    /// let c = rindex.insert([3.0, 3.0]);
    /// let d = rindex.insert([20.0, 20.0]);
    ///
    /// let query_point = [0.0, 0.0];
    /// let query_radius = 10.0;
    /// let (neighbors, distances) = rindex.query(&query_point, query_radius);
    /// assert_eq!(neighbors, vec![a, b, c]);
    /// ```
    #[must_use]
    pub fn query(&self, point: &[f64; D], radius: f64) -> (Vec<usize>, Vec<f64>) {
        let mut result = Vec::new();
        let mut queue = vec![self.root];
        while let Some(node_id) = queue.pop() {
            let node = &self.index.nodes[node_id];
            if node.is_leaf() {
                for &child_id in &node.children {
                    let child = &self.index.nodes[child_id];
                    let distance = child.sphere.min_distance(point);
                    if distance <= radius {
                        result.push((child_id, distance));
                    }
                }
            } else {
                for &child_id in &node.children {
                    let child = &self.index.nodes[child_id];
                    let distance = child.sphere.min_distance(point);
                    if distance <= radius {
                        queue.push(child_id);
                    }
                }
            }
        }
        result.sort_by_key(|(_, dist)| OrderedFloat(*dist));
        let indices = result.iter().map(|(id, _)| *id).collect();
        let distances = result.iter().map(|(_, dist)| *dist).collect();
        (indices, distances)
    }

    /// # Examples
    /// ```
    /// use rindex::Rindex;
    ///
    /// let mut rindex = Rindex::default();
    /// let a = rindex.insert([1.0, 1.0]);
    /// let b = rindex.insert([2.0, 2.0]);
    /// let c = rindex.insert([3.0, 3.0]);
    /// let d = rindex.insert([20.0, 20.0]);
    ///
    /// let query_point = [0.0, 0.0];
    /// let (neighbors, distances) = rindex.query_neighbors(&query_point, 3);
    /// assert_eq!(neighbors, vec![a, b, c]);
    /// ```
    #[must_use]
    pub fn query_neighbors(&self, point: &[f64; D], k: usize) -> (Vec<usize>, Vec<f64>) {
        if k == 0 || self.root == usize::MAX {
            return (Vec::new(), Vec::new());
        }
        let mut result = BinaryHeap::from(vec![(OrderedFloat(f64::INFINITY), usize::MAX); k]);
        self.query_neighbors_recursive(self.root, point, &mut result);
        let mut indices = Vec::with_capacity(k);
        let mut distances = Vec::with_capacity(k);
        while let Some((distance, id)) = result.pop() {
            if id != usize::MAX {
                indices.push(id);
                distances.push(distance.into_inner());
            }
        }
        indices.reverse();
        distances.reverse();
        (indices, distances)
    }

    /// # Examples
    /// ```
    /// use rindex::Rindex;
    ///
    /// let k = 3;
    /// let mut rindex = Rindex::new(k);
    /// let a = rindex.insert([1.0, 1.0]);
    /// let b = rindex.insert([2.0, 2.0]);
    /// let c = rindex.insert([3.0, 3.0]);
    /// let d = rindex.insert([20.0, 20.0]);
    ///
    /// let (neighbors, distances) = rindex.query_reverse(&[0.0, 0.0]);
    /// // Points a sees the query point as one of its k nearest neighbors
    /// assert_eq!(neighbors, vec![a]);
    #[must_use]
    pub fn query_reverse(&self, point: &[f64; D]) -> (Vec<usize>, Vec<f64>) {
        if self.k == 0 || self.root == usize::MAX {
            return (Vec::new(), Vec::new());
        }
        let mut neighbors = Vec::new();
        self.query_reverse_recursive(self.root, point, &mut neighbors);
        neighbors.sort_by_key(|(_, dist)| OrderedFloat(*dist));
        let indices = neighbors.iter().map(|(id, _)| *id).collect();
        let distances = neighbors.iter().map(|(_, dist)| *dist).collect();
        (indices, distances)
    }

    /// # Examples
    /// ```
    /// use rindex::Rindex;
    ///
    /// let mut rindex = Rindex::default();
    /// let a = rindex.insert([1.0, 1.0]); // returns an id of the inserted point
    /// assert_eq!(rindex.point_at(a), &[1.0, 1.0]);
    #[must_use]
    pub fn point_at(&self, slot_id: usize) -> &[f64; D] {
        &self.index.nodes[slot_id].sphere.center
    }

    /// # Examples
    /// ```
    /// use rindex::Rindex;
    ///
    /// let k = 3;
    /// let mut rindex = Rindex::new(k);
    ///
    /// // Insert some points
    /// let a = rindex.insert([1.0, 1.0]);
    /// let b = rindex.insert([2.0, 2.0]);
    /// let c = rindex.insert([3.0, 3.0]);
    /// let d = rindex.insert([20.0, 20.0]);
    ///
    /// // Check the k nearest neighbors of point a
    /// let (neighbors, distances) = rindex.neighbors_of(a);
    /// assert_eq!(neighbors, vec![a, b, c]);
    #[must_use]
    pub fn neighbors_of(&self, point_id: usize) -> (Vec<usize>, Vec<f64>) {
        let neighbors = &self.index.nodes[point_id].neighbors;
        let mut neighbors: Vec<(OrderedFloat<f64>, usize)> =
            neighbors.iter().map(|(dist, id)| (*dist, *id)).collect();
        neighbors.sort_by_key(|(dist, _)| OrderedFloat(*dist));

        // Remove the dummy neighbors with infinite distances
        let neighbors: Vec<(OrderedFloat<f64>, usize)> = neighbors
            .iter()
            .filter(|(dist, _)| dist.0 != f64::INFINITY)
            .map(|(dist, id)| (*dist, *id))
            .collect();

        let indices = neighbors.iter().map(|(_, id)| *id).collect();
        let distances = neighbors.iter().map(|(dist, _)| dist.0).collect();
        (indices, distances)
    }

    /// # Examples
    /// ```
    /// use rindex::Rindex;
    ///
    /// let k = 3;
    /// let mut rindex = Rindex::new(k);
    ///
    /// // Insert some points
    /// let a = rindex.insert([1.0, 1.0]);
    /// let b = rindex.insert([2.0, 2.0]);
    /// let c = rindex.insert([3.0, 3.0]);
    /// let d = rindex.insert([20.0, 20.0]);
    ///
    /// // Check the distance to the k-th nearest neighbor of point a
    /// assert_eq!(rindex.knn_dist_of(a), 8_f64.sqrt()); // distance to point c
    #[must_use]
    pub fn knn_dist_of(&self, point_id: usize) -> f64 {
        self.index.nodes[point_id]
            .neighbors
            .peek()
            .unwrap_or(&(OrderedFloat(f64::INFINITY), usize::MAX))
            .0
             .0
    }

    /// # Examples
    /// ```
    /// use rindex::Rindex;
    ///
    /// let fanout = 4;
    /// let k = 3;
    /// let mut rindex = Rindex::new_with_params(fanout, k).expect("Failed to create Rindex");
    ///
    /// // Insert some points
    /// let a = rindex.insert([1.0, 1.0]);
    /// let b = rindex.insert([2.0, 2.0]);
    /// let c = rindex.insert([3.0, 3.0]);
    /// let d = rindex.insert([20.0, 20.0]);
    ///
    /// // Check the height of the tree
    /// assert_eq!(rindex.height(), 1); // since the fanout is 4
    ///
    /// // One more insertion should increase the height
    /// let e = rindex.insert([30.0, 30.0]);
    /// assert_eq!(rindex.height(), 2);
    #[must_use]
    pub fn height(&self) -> usize {
        self.index
            .nodes
            .get(self.root)
            .map_or(0, |node| node.height)
    }

    #[must_use]
    pub fn num_points(&self) -> usize {
        self.num_points
    }

    #[must_use]
    pub fn nodes_to_string_rows(&self) -> Vec<String> {
        let mut rows = Vec::new();
        let height = self.height();
        for h in (0..=height).rev() {
            for node in &self.index.nodes {
                if node.height == h && !node.is_deleted() {
                    rows.push(node.to_string());
                }
            }
        }
        rows
    }
}

// Update algorithms
impl<const D: usize> Rindex<D> {
    fn reinsert_nodes(&mut self, reinsert_list: &mut Vec<usize>) {
        self.reinsert_height = 1;
        while let Some(entry_id) = reinsert_list.pop() {
            let res = self.insert_entry(self.root, entry_id);
            reinsert_list.extend(res);
            self.reinsert_height += 1;
        }
        self.adjust_tree();
    }

    // Insert a node into the tree
    // Split the node if it has too many children
    fn insert_entry(&mut self, node: usize, entry: usize) -> Vec<usize> {
        if self.index.nodes[node].height == self.index.nodes[entry].height + 1 {
            self.index.nodes[node].children.push(entry);
            self.reshape(node);

            let mut to_be_reinserted = Vec::new();
            if self.index.nodes[node].children.len() > self.max_fanout && node != self.root {
                if self.reinsert_height == self.index.nodes[node].height && self.reinsert_fanout > 0
                {
                    to_be_reinserted.extend(self.pop_farthest_children(node));
                } else {
                    to_be_reinserted.push(self.split(node));
                }
            }
            to_be_reinserted
        } else {
            let best_child = self.choose_subtree(node, entry);
            let result = self.insert_entry(best_child, entry);
            self.reshape(node);
            result
        }
    }

    fn delete_entry(&mut self, entry: usize) -> Vec<usize> {
        let mut current = self.index.nodes[entry].parent;
        self.index.nodes[current].children.retain(|&x| x != entry);
        self.delete_slot(entry);

        let mut reinsert_list = Vec::new();
        while current != usize::MAX {
            self.reshape(current);
            let parent = self.index.nodes[current].parent;
            if current != self.root && self.index.nodes[current].children.len() < self.min_fanout {
                self.index.nodes[parent].children.retain(|&x| x != current);
                reinsert_list.extend(self.index.nodes[current].children.clone());
                self.delete_slot(current);
            }
            current = parent;
        }
        reinsert_list
    }

    fn adjust_tree(&mut self) {
        self.reshape(self.root);
        if self.index.nodes[self.root].children.len() > self.max_fanout {
            let sibling = self.split(self.root);
            let new_root = Node::internal(vec![self.root, sibling]);
            self.root = self.add_slot(new_root);
            self.reshape(self.root);
        } else if self.index.nodes[self.root].height > 1
            && self.index.nodes[self.root].children.len() == 1
        {
            let new_root = self.index.nodes[self.root].children[0];
            self.delete_slot(self.root);
            self.root = new_root;
            self.index.nodes[self.root].parent = usize::MAX;
        } else if self.index.nodes[self.root].children.is_empty() {
            self.delete_slot(self.root);
            self.root = usize::MAX;
        }
    }

    fn choose_subtree(&self, node: usize, entry: usize) -> usize {
        let mut best_distance = f64::INFINITY;
        let mut best_child = usize::MAX;
        for &child_id in &self.index.nodes[node].children {
            let child = &self.index.nodes[child_id];
            let distance = euclidean(&child.sphere.center, &self.index.nodes[entry].sphere.center);
            if distance < best_distance {
                best_distance = distance;
                best_child = child_id;
            }
        }
        best_child
    }

    #[allow(clippy::similar_names)]
    fn reshape(&mut self, slot_id: usize) {
        // Calculate the sphere
        let mut sphere = self.calculate_sphere(&self.index.nodes[slot_id].children);
        sphere.variance = self.calculate_variance(slot_id);
        self.index.nodes[slot_id].sphere = sphere;

        // Calculate the height
        self.index.nodes[slot_id].height = self.index.nodes[slot_id]
            .children
            .iter()
            .fold(0, |max, child_id| {
                max.max(self.index.nodes[*child_id].height)
            })
            + 1;

        // Update parent of the children
        for child_id in self.index.nodes[slot_id].children.clone() {
            self.index.nodes[child_id].parent = slot_id;
        }
    }

    fn add_slot(&mut self, node: Node<D>) -> usize {
        let slot_id = self.index.insert(node);
        if self.index.nodes[slot_id].is_point() {
            let mut neighbors =
                BinaryHeap::from(vec![(OrderedFloat(f64::INFINITY), usize::MAX); self.k - 1]);
            neighbors.push((OrderedFloat(0.0), slot_id));
            self.preinsert(self.root, slot_id, &mut neighbors);
            self.index.nodes[slot_id].neighbors = neighbors;
        }
        slot_id
    }

    fn delete_slot(&mut self, slot_id: usize) {
        self.index.delete(slot_id);
        self.index.nodes[slot_id] = Node::default();
    }

    fn update_node_bound(&mut self, node_id: usize) {
        let mut bound: f64 = 0.0;
        for child_id in &self.index.nodes[node_id].children {
            bound = bound.max(self.index.nodes[*child_id].bound());
        }
        self.index.nodes[node_id].sphere.bound = bound;
    }

    fn preinsert(
        &mut self,
        node_id: usize,
        point_id: usize,
        neighbors: &mut BinaryHeap<(OrderedFloat<f64>, usize)>,
    ) {
        if node_id == usize::MAX {
            return;
        }
        if self.index.nodes[node_id].is_leaf() {
            let children = self.index.nodes[node_id].children.clone();
            for child in children {
                let distance = self.index.nodes[child]
                    .sphere
                    .min_distance(&self.index.nodes[point_id].sphere.center);
                let current_kth = neighbors
                    .peek()
                    .unwrap_or(&(OrderedFloat(f64::INFINITY), 0))
                    .0;
                if distance < current_kth.0 {
                    neighbors.pop();
                    neighbors.push((OrderedFloat(distance), child));
                }

                let neighbor_kth = self.index.nodes[child]
                    .neighbors
                    .peek()
                    .unwrap_or(&(OrderedFloat(f64::INFINITY), 0))
                    .0;
                if distance < neighbor_kth.0 {
                    self.index.nodes[child].neighbors.pop();
                    self.index.nodes[child]
                        .neighbors
                        .push((OrderedFloat(distance), point_id));
                }
            }
        } else {
            let mut children = self.index.nodes[node_id]
                .children
                .iter()
                .map(|&child_id| {
                    let child = &self.index.nodes[child_id];
                    let distance = child.min_distance(&self.index.nodes[point_id].sphere.center);
                    (OrderedFloat(distance), child_id)
                })
                .collect::<Vec<_>>();
            children.sort_by_key(|x| x.0);

            for (distance, child_id) in children {
                let current_kth = neighbors
                    .peek()
                    .unwrap_or(&(OrderedFloat(f64::INFINITY), 0))
                    .0;
                let max_knn = OrderedFloat(self.index.nodes[child_id].sphere.bound);
                if distance <= max_knn || distance <= current_kth {
                    self.preinsert(child_id, point_id, neighbors);
                }
            }
        }
        self.update_node_bound(node_id);
    }

    fn post_delete(&mut self, node_id: usize, points: &Vec<usize>, sphere: &Sphere<D>) {
        if node_id == usize::MAX {
            return;
        }
        let node = &self.index.nodes[node_id];
        if node.is_leaf() {
            let children = node.children.clone();
            for point in points {
                for child in &children {
                    let distance = self.index.nodes[*child]
                        .min_distance(&self.index.nodes[*point].sphere.center);
                    let current_kth = self.index.nodes[*point]
                        .neighbors
                        .peek()
                        .unwrap_or(&(OrderedFloat(f64::INFINITY), 0))
                        .0;
                    if distance < current_kth.0 {
                        self.index.nodes[*point].neighbors.pop();
                        self.index.nodes[*point]
                            .neighbors
                            .push((OrderedFloat(distance), *child));
                    }
                }
            }
        } else {
            let mut children = node
                .children
                .iter()
                .map(|&child_id| {
                    let child = &self.index.nodes[child_id];
                    let distance = (child.min_distance(&sphere.center) - sphere.radius).max(0.0);
                    (OrderedFloat(distance), child_id)
                })
                .collect::<Vec<_>>();
            children.sort_by_key(|x| x.0);

            for (distance, child_id) in children {
                let mut max_current_kth = OrderedFloat(0.0);
                for point in points {
                    let current_kth = self.index.nodes[*point]
                        .neighbors
                        .peek()
                        .unwrap_or(&(OrderedFloat(f64::INFINITY), 0))
                        .0;
                    max_current_kth = max_current_kth.max(current_kth);
                }
                if distance >= max_current_kth {
                    break;
                }
                self.post_delete(child_id, points, sphere);
            }
        }
        self.update_node_bound(node_id);
    }
}

// Split algorithms:
// - pop_farthest_children: Pop the children with the farthest distance from the parent
// - split: split the node along the dimension with the maximum variance
// - split_dimension: Find the dimension with the maximum variance
impl<const D: usize> Rindex<D> {
    fn pop_farthest_children(&mut self, node: usize) -> Vec<usize> {
        let mut children = self.index.nodes[node].children.clone();
        children.sort_by_key(|child| {
            let child_sphere = &self.index.nodes[*child].sphere;
            let dist = child_sphere.max_distance(&self.index.nodes[node].sphere.center);
            OrderedFloat(dist + child_sphere.radius)
        });
        let to_be_reinserted = children.split_off(children.len() - self.reinsert_fanout);
        self.index.nodes[node].children = children;
        self.reshape(node);
        to_be_reinserted
    }

    fn split(&mut self, slot_id: usize) -> usize {
        // Find the split dimension
        let split_dimension = self.split_dimension(slot_id);

        // Sort the children along the split dimension
        let mut left = self.index.nodes[slot_id].children.clone();
        left.sort_by(|a, b| {
            let a = &self.index.nodes[*a].sphere.center[split_dimension];
            let b = &self.index.nodes[*b].sphere.center[split_dimension];
            a.partial_cmp(b).unwrap()
        });

        // Split the children into two groups
        let mut right = left.split_off(self.min_fanout);
        right.reverse();
        let mut remaining = right.split_off(self.min_fanout);

        assert_eq!(left.len(), self.min_fanout);
        assert_eq!(right.len(), self.min_fanout);

        let left_sphere = self.calculate_sphere(&left);
        let right_sphere = self.calculate_sphere(&right);
        let left_dist = euclidean(
            &left_sphere.center,
            &self.index.nodes[slot_id].sphere.center,
        );
        let right_dist = euclidean(
            &right_sphere.center,
            &self.index.nodes[slot_id].sphere.center,
        );

        if right_dist < left_dist {
            std::mem::swap(&mut left, &mut right);
        }

        // Create two new nodes
        self.index.nodes[slot_id].children = left;
        self.reshape(slot_id);

        let sibling = Node::internal(right);
        let sibling = self.add_slot(sibling);
        self.reshape(sibling);

        // Reinsert the remaining children
        while let Some(entry) = remaining.pop() {
            let node_dist = euclidean(
                &self.index.nodes[entry].sphere.center,
                &self.index.nodes[slot_id].sphere.center,
            );
            let sibling_dist = euclidean(
                &self.index.nodes[entry].sphere.center,
                &self.index.nodes[sibling].sphere.center,
            );
            if node_dist < sibling_dist {
                self.index.nodes[slot_id].children.push(entry);
            } else {
                self.index.nodes[sibling].children.push(entry);
            }
        }

        // Finalize the split
        self.reshape(slot_id);
        self.reshape(sibling);

        sibling
    }

    fn split_dimension(&self, slot_id: usize) -> usize {
        // Calculate the variance at each dimension
        let variance = self.calculate_variance(slot_id);

        // Find the dimension with the maximum variance
        variance
            .iter()
            .enumerate()
            .max_by_key(|(_, &variance)| OrderedFloat(variance))
            .map_or(0, |(i, _)| i)
    }

    fn calculate_variance(&self, slot_id: usize) -> [f64; D] {
        let node = &self.index.nodes[slot_id];
        let mean = &node.sphere.center;
        let mut variance = [0.0; D];
        for child_id in &node.children {
            let child = &self.index.nodes[*child_id].sphere;
            for (i, x) in child.center.iter().enumerate() {
                variance[i] += (x - mean[i]).powi(2) * child.weight;
                variance[i] += child.variance[i] * child.weight;
            }
        }
        for x in &mut variance {
            *x /= node.sphere.weight;
        }
        variance
    }

    fn calculate_sphere(&self, children: &[usize]) -> Sphere<D> {
        // Calculate the centroid
        let mut centroid = [0.0; D];
        let mut weight = 0.0;
        for child_id in children {
            let child = &self.index.nodes[*child_id].sphere;
            for (i, x) in child.center.iter().enumerate() {
                centroid[i] += x * child.weight;
            }
            weight += child.weight;
        }
        for x in &mut centroid {
            *x /= weight;
        }

        // Calculate the radius & bound
        let mut radius: f64 = 0.0;
        let mut bound: f64 = 0.0;
        for child_id in children {
            let child = &self.index.nodes[*child_id];
            let distance = child.max_distance(&centroid);
            radius = radius.max(distance);
            bound = bound.max(child.bound());
        }
        let mut sphere = Sphere::new(centroid, radius, weight);
        sphere.bound = bound;
        sphere
    }
}

// Query algorithms
impl<const D: usize> Rindex<D> {
    fn query_neighbors_recursive(
        &self,
        node: usize,
        point: &[f64; D],
        neighbors: &mut BinaryHeap<(OrderedFloat<f64>, usize)>,
    ) {
        let node = &self.index.nodes[node];
        if node.is_leaf() {
            for child in &node.children {
                let child = &self.index.nodes[*child];
                let distance = child.sphere.min_distance(point);
                let current_kth = neighbors
                    .peek()
                    .unwrap_or(&(OrderedFloat(f64::INFINITY), 0))
                    .0;
                if distance < current_kth.0 {
                    neighbors.pop();
                    neighbors.push((OrderedFloat(distance), child.slot_id));
                }
            }
        } else {
            let mut children = node
                .children
                .iter()
                .map(|&child_id| {
                    let child = &self.index.nodes[child_id];
                    let distance = child.min_distance(point);
                    (OrderedFloat(distance), child_id)
                })
                .collect::<Vec<_>>();
            children.sort_by_key(|x| x.0);

            for (distance, child_id) in children {
                let current_kth = neighbors
                    .peek()
                    .unwrap_or(&(OrderedFloat(f64::INFINITY), 0))
                    .0;
                if distance >= current_kth {
                    break;
                }
                self.query_neighbors_recursive(child_id, point, neighbors);
            }
        }
    }

    fn query_reverse_recursive(
        &self,
        node: usize,
        point: &[f64; D],
        reverse_neighbors: &mut Vec<(usize, f64)>,
    ) {
        let node = &self.index.nodes[node];
        if node.is_leaf() {
            for child in &node.children {
                let child = &self.index.nodes[*child];
                let distance = child.sphere.min_distance(point);
                if distance <= child.bound() {
                    reverse_neighbors.push((child.slot_id, distance));
                }
            }
        } else {
            for child in &node.children {
                let distance = self.index.nodes[*child].min_distance(point);
                if distance > self.index.nodes[*child].bound() {
                    continue;
                }
                self.query_reverse_recursive(*child, point, reverse_neighbors);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::Rindex;
    use crate::{distance::euclidean, node::Node};

    #[test]
    fn reshape() {
        let mut rindex = Rindex::default();

        // Create a parent node
        let parent = rindex.add_slot(Node::leaf());

        // Create some point nodes
        let node_a = rindex.add_slot(Node::point([0.0, 0.0]));
        let node_b = rindex.add_slot(Node::point([0.0, 2.0]));
        let node_c = rindex.add_slot(Node::point([2.0, 0.0]));
        let node_d = rindex.add_slot(Node::point([2.0, 2.0]));
        rindex.index.nodes[parent].children = vec![node_a, node_b, node_c, node_d];

        // Reshape the parent node
        rindex.reshape(parent);

        // Check the parent node's sphere
        let sphere = &rindex.index.nodes[parent].sphere;
        assert_eq!(sphere.center, [1., 1.]);
        assert_eq!(sphere.radius, 2.0_f64.sqrt());
        assert_eq!(sphere.weight, 4.0);

        // Check the parent-child relationship
        assert_eq!(rindex.index.nodes[node_a].parent, parent);
        assert_eq!(rindex.index.nodes[node_b].parent, parent);
        assert_eq!(rindex.index.nodes[node_c].parent, parent);
        assert_eq!(rindex.index.nodes[node_d].parent, parent);
    }

    #[test]
    fn split() {
        let fanout = 8;
        let k = 10;
        let mut rindex = Rindex::new_with_params(fanout, k).expect("Invalid fanout");

        // Create 9 point nodes, as the fanout of 8 will trigger a split
        let node_a = rindex.add_slot(Node::point([0.0, 0.0]));
        let node_b = rindex.add_slot(Node::point([0.0, 2.0]));
        let node_c = rindex.add_slot(Node::point([1.0, 1.0]));
        let node_d = rindex.add_slot(Node::point([2.0, 0.0]));
        let node_e = rindex.add_slot(Node::point([2.0, 2.0]));
        let node_w = rindex.add_slot(Node::point([10.0, 1.0]));
        let node_x = rindex.add_slot(Node::point([10.0, 2.0]));
        let node_y = rindex.add_slot(Node::point([11.0, 1.0]));
        let node_z = rindex.add_slot(Node::point([11.0, 2.0]));
        let point_nodes = vec![
            node_a, node_b, node_c, node_d, node_e, node_w, node_x, node_y, node_z,
        ];

        // Create a node
        let node = Node::internal(point_nodes.clone());
        let node = rindex.add_slot(node);
        rindex.reshape(node);

        // Check the split dimension
        let dimension = rindex.split_dimension(node);
        assert_eq!(dimension, 0);

        // Split the parent node
        let sibling = rindex.split(node);

        // Check the node's sphere
        let node = &rindex.index.nodes[node];
        assert_eq!(node.sphere.center, [1., 1.]);
        assert_eq!(node.sphere.radius, 2.0_f64.sqrt());
        assert_eq!(node.sphere.weight, 5.0);

        // Check the sibling's sphere
        let sibling = &rindex.index.nodes[sibling];
        assert_eq!(sibling.sphere.center, [10.5, 1.5]);
        assert_eq!(
            sibling.sphere.radius,
            (0.5_f64.powi(2) + 0.5_f64.powi(2)).sqrt()
        );
        assert_eq!(sibling.sphere.weight, 4.0);
    }

    #[test]
    fn update() {
        let fanout = 8;
        let k = 10;
        let mut rindex = Rindex::new_with_params(fanout, k).expect("Invalid fanout");

        // The tree should be empty
        assert_eq!(rindex.height(), 0);

        // Insert fanout points to fill the root node
        let mut point_ids = Vec::new();
        for i in 0..fanout {
            let point_id = rindex.insert([i as f64, i as f64]);
            point_ids.push(point_id);
        }

        // The tree should be of height 1 since
        // the number of points is equal to the fanout
        assert_eq!(rindex.height(), 1);

        // Insert one more point to trigger a split (so the tree grows in height)
        let last_inserted = rindex.insert([fanout as f64, fanout as f64]);
        point_ids.push(last_inserted);
        assert_eq!(rindex.height(), 2);

        // Delete two points to trigger a merge (so the tree shrinks in height)
        rindex.delete(point_ids.pop().unwrap());
        rindex.delete(point_ids.pop().unwrap());

        // The tree should be of height 1 again
        assert_eq!(rindex.height(), 1);

        // Delete all remaining points
        for point_id in point_ids {
            rindex.delete(point_id);
        }

        // The tree should be empty again
        assert_eq!(rindex.height(), 0);
    }

    #[test]
    fn verify_fanout_params() {
        let mut rindex = Rindex::default();
        let n = 1000;
        let mut rng = StdRng::seed_from_u64(0);
        let deletion_probability = 0.2;

        // Perform a random sequence of insertions and deletions
        let mut point_ids = Vec::new();
        for _ in 0..n {
            let should_delete = rng.gen_bool(deletion_probability);
            if should_delete && !point_ids.is_empty() {
                let random_index = rng.gen_range(0..point_ids.len());
                let point_id = point_ids.swap_remove(random_index);
                rindex.delete(point_id);
            } else {
                let point = [rng.gen_range(0.0..100.0), rng.gen_range(0.0..100.0)];
                let point_id = rindex.insert(point);
                point_ids.push(point_id);
            }

            // Check the fanout constraints after each operation
            for node in &rindex.index.nodes {
                if !node.is_point() && node.slot_id != rindex.root && !node.is_deleted() {
                    assert!(node.children.len() >= rindex.min_fanout);
                    assert!(node.children.len() <= rindex.max_fanout);
                }
            }
        }
    }

    #[test]
    fn knn_distances() {
        let fanout = 5;
        let k = 5;
        let mut rindex = Rindex::new_with_params(fanout, k).expect("Invalid fanout");

        // Insert some points
        let a = rindex.insert([0.0, 1.0]);
        let b = rindex.insert([0.0, 2.0]);
        let c = rindex.insert([0.0, 3.0]);
        let d = rindex.insert([0.0, 4.0]);
        let e = rindex.insert([0.0, 5.0]);

        // Confirms that knn distances are updated after inserting a new point
        assert_eq!(rindex.knn_dist_of(a), 4.0);
        assert_eq!(rindex.knn_dist_of(b), 3.0);
        assert_eq!(rindex.knn_dist_of(c), 2.0);
        assert_eq!(rindex.knn_dist_of(d), 3.0);
        assert_eq!(rindex.knn_dist_of(e), 4.0);

        // We insert a new point that is closer to a than the current farthest neighbor
        let f = rindex.insert([0.0, 6.0]);
        assert_eq!(rindex.knn_dist_of(a), 4.0);
        assert_eq!(rindex.knn_dist_of(b), 3.0);
        assert_eq!(rindex.knn_dist_of(c), 2.0);
        assert_eq!(rindex.knn_dist_of(d), 2.0);
        assert_eq!(rindex.knn_dist_of(e), 3.0);
        assert_eq!(rindex.knn_dist_of(f), 4.0);

        // Delete the point a and check the neighbors of the remaining points
        rindex.delete(a);

        assert_eq!(rindex.knn_dist_of(b), 4.0);
        assert_eq!(rindex.knn_dist_of(c), 3.0);
        assert_eq!(rindex.knn_dist_of(d), 2.0);
        assert_eq!(rindex.knn_dist_of(e), 3.0);
        assert_eq!(rindex.knn_dist_of(f), 4.0);
    }

    #[test]
    fn query() {
        let mut rindex = Rindex::default();

        // Insert some points
        let mut point_ids = Vec::new();
        for i in 0..100 {
            let point_id = rindex.insert([i as f64, i as f64]);
            point_ids.push(point_id);
        }

        // Set the query point to the center of the data layout
        let query_point = [50.0, 50.0];
        let query_radius = 5.0;

        // Find the expected points within the radius
        let mut expected = Vec::new();
        for p in point_ids {
            let point = rindex.index.nodes[p].sphere.center;
            let distance = euclidean(&point, &query_point);
            if distance <= query_radius {
                expected.push(p);
            }
        }

        // Query the tree for the points within the radius
        let (mut range_query_result, _) = rindex.query(&query_point, query_radius);
        range_query_result.sort();
        assert_eq!(expected, range_query_result);

        // Query the tree for k nearest neighbors of the query point
        let (mut knn_query_result, _) =
            rindex.query_neighbors(&query_point, range_query_result.len());
        knn_query_result.sort();

        // The results of the range query and the kNN query should be the same
        assert_eq!(expected, knn_query_result);
    }

    #[test]
    fn reverse_query() {
        let k = 5;
        let mut rindex = Rindex::new(k);

        for i in 0..100 {
            let _ = rindex.insert([i as f64, i as f64]);
        }

        // Query the tree in the middle of the data layout
        let query_point = [50.0, 50.0];
        let (_, distances) = rindex.query_reverse(&query_point);
        assert_eq!(distances.len(), 5);
        assert_eq!(distances[0], 0.0);
        assert_eq!(distances[1], 2_f64.sqrt());
        assert_eq!(distances[2], 2_f64.sqrt());
        assert_eq!(distances[3], 8_f64.sqrt());
        assert_eq!(distances[4], 8_f64.sqrt());
    }
}
