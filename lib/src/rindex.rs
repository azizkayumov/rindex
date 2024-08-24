use crate::{distance::euclidean, index::Index, node::Node};
use ordered_float::OrderedFloat;

pub struct Rindex<const D: usize> {
    min_fanout: usize,
    max_fanout: usize,
    reinsert_fanout: usize,
    reinsert_height: usize,
    root: usize,
    nodes: Vec<Node<D>>,
    index: Index,
}

impl<const D: usize> Rindex<D> {
    #[must_use]
    pub fn new(fanout: usize) -> Option<Self> {
        if fanout < 4 {
            return None;
        }
        Some(Rindex {
            min_fanout: fanout / 2,
            max_fanout: fanout,
            reinsert_fanout: fanout / 3,
            reinsert_height: 1,
            root: usize::MAX,
            nodes: Vec::new(),
            index: Index::new(),
        })
    }

    #[must_use]
    pub fn insert(&mut self, point: [f64; D]) -> usize {
        // Create the root node if it doesn't exist
        if self.root == usize::MAX {
            let node = Node::leaf();
            self.root = self.add_slot(node);
        }

        // Create a point node (reuse a slot in the node vector if possible)
        let point = Node::point(point);
        let slot_id = self.add_slot(point);

        // Insert the point node into the tree
        let mut reinsert_list = vec![slot_id];
        self.reinsert_height = 1;
        self.reinsert_nodes(&mut reinsert_list);

        slot_id
    }

    pub fn delete(&mut self, point_id: usize) {
        let mut reinsert_list = self.delete_entry(point_id);
        self.reinsert_height = 1;
        self.reinsert_nodes(&mut reinsert_list);
    }

    #[must_use]
    pub fn query(&self, _point: [f64; D], _radius: f64) -> Vec<usize> {
        todo!("Implement query");
    }

    #[must_use]
    pub fn query_neighbors(&self, _point: [f64; D], _k: usize) -> Vec<usize> {
        todo!("Implement query_nearest");
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.nodes.get(self.root).map_or(0, |node| node.height)
    }

    fn reinsert_nodes(&mut self, reinsert_list: &mut Vec<usize>) {
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
        if self.nodes[node].height == self.nodes[entry].height + 1 {
            self.nodes[node].children.push(entry);
            self.reshape(node);

            let mut to_be_reinserted = Vec::new();
            if self.nodes[node].children.len() > self.max_fanout && node != self.root {
                if self.reinsert_height == self.nodes[node].height && self.reinsert_fanout > 0 {
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
        let mut current = self.nodes[entry].parent;
        self.nodes[current].children.retain(|&x| x != entry);
        self.delete_slot(entry);

        let mut reinsert_list = Vec::new();
        while current != usize::MAX {
            self.reshape(current);
            let parent = self.nodes[current].parent;
            if current != self.root && self.nodes[current].children.len() < self.min_fanout {
                self.nodes[parent].children.retain(|&x| x != current);
                reinsert_list.extend(self.nodes[current].children.clone());
                self.delete_slot(current);
            }
            current = parent;
        }
        reinsert_list
    }

    fn adjust_tree(&mut self) {
        self.reshape(self.root);
        if self.nodes[self.root].children.len() > self.max_fanout {
            let sibling = self.split(self.root);
            let new_root = Node::internal(vec![self.root, sibling]);
            self.root = self.add_slot(new_root);
            self.reshape(self.root);
        } else if self.nodes[self.root].height > 1 && self.nodes[self.root].children.len() == 1 {
            let new_root = self.nodes[self.root].children[0];
            self.delete_slot(self.root);
            self.root = new_root;
            self.nodes[self.root].parent = usize::MAX;
        } else if self.nodes[self.root].children.is_empty() {
            self.delete_slot(self.root);
            self.root = usize::MAX;
        }
    }

    fn choose_subtree(&self, node: usize, entry: usize) -> usize {
        let mut best_distance = f64::INFINITY;
        let mut best_child = usize::MAX;
        for &child_id in &self.nodes[node].children {
            let child = &self.nodes[child_id];
            let distance = euclidean(&child.sphere.center, &self.nodes[entry].sphere.center);
            if distance < best_distance {
                best_distance = distance;
                best_child = child_id;
            }
        }
        best_child
    }

    fn pop_farthest_children(&mut self, node: usize) -> Vec<usize> {
        let mut children = self.nodes[node].children.clone();
        children.sort_by_key(|child| {
            let child_sphere = self.nodes[*child].sphere;
            let dist = euclidean(&self.nodes[node].sphere.center, &child_sphere.center);
            OrderedFloat(dist + child_sphere.radius)
        });
        let to_be_reinserted = children.split_off(children.len() - self.reinsert_fanout);
        self.nodes[node].children = children;
        self.reshape(node);
        to_be_reinserted
    }

    fn split(&mut self, slot_id: usize) -> usize {
        // Find the farthest child from the centroid as the sibling seed
        let mut sibling_seed: usize = self.nodes[slot_id].children[0];
        let mut max_dist = 0.0;
        for child in &self.nodes[slot_id].children {
            let child_sphere = &self.nodes[*child].sphere;
            let distance = euclidean(&self.nodes[slot_id].sphere.center, &child_sphere.center)
                + child_sphere.radius;
            if distance > max_dist {
                sibling_seed = *child;
                max_dist = distance;
            }
        }

        // Give minimum fanout children to the sibling
        let mut children = self.nodes[slot_id].children.clone();
        children.sort_by_key(|child| {
            let child_sphere = self.nodes[*child].sphere;
            let distance = euclidean(
                &self.nodes[sibling_seed].sphere.center,
                &child_sphere.center,
            ) + child_sphere.radius;
            OrderedFloat(distance)
        });
        let mut remaining = children.split_off(self.min_fanout);

        // Both nodes should have at least min_fanout children
        let sibling = self.add_slot(Node::internal(children));
        self.reshape(sibling);
        self.nodes[slot_id].children = remaining.split_off(remaining.len() - self.min_fanout);
        self.reshape(slot_id);

        // Distribute the remaining children to whichever node is closer
        for r in remaining {
            let dist_sibling = euclidean(
                &self.nodes[sibling].sphere.center,
                &self.nodes[r].sphere.center,
            );
            let dist_node = euclidean(
                &self.nodes[slot_id].sphere.center,
                &self.nodes[r].sphere.center,
            );
            if dist_sibling < dist_node {
                self.nodes[sibling].children.push(r);
            } else {
                self.nodes[slot_id].children.push(r);
            }
        }

        // Finally, reshape both nodes
        self.reshape(sibling);
        self.reshape(slot_id);

        sibling
    }

    #[allow(clippy::similar_names)]
    fn reshape(&mut self, slot_id: usize) {
        // Calculate the centroid, weight and height of the parent
        let mut centroid = [0.0; D];
        let mut weight = 0.0;
        let mut height = 0;
        for child_id in &self.nodes[slot_id].children {
            let child = &self.nodes[*child_id].sphere;
            for (i, x) in child.center.iter().enumerate() {
                centroid[i] += x * child.weight;
            }
            weight += child.weight;
            height = height.max(self.nodes[*child_id].height);
        }
        for x in &mut centroid {
            *x /= weight;
        }

        // Calculate the radius of the new sphere
        let mut radius: f64 = 0.0;
        for child_id in &self.nodes[slot_id].children {
            let child = &self.nodes[*child_id].sphere;
            let distance = euclidean(&centroid, &child.center);
            radius = radius.max(distance + child.radius);
        }

        // Update the sphere & height of the parent
        self.nodes[slot_id].sphere.center = centroid;
        self.nodes[slot_id].sphere.radius = radius;
        self.nodes[slot_id].sphere.weight = weight;
        self.nodes[slot_id].height = height + 1;

        // Update parent of the children
        for child_id in self.nodes[slot_id].children.clone() {
            self.nodes[child_id].parent = slot_id;
        }
    }

    fn add_slot(&mut self, mut node: Node<D>) -> usize {
        let slot_id = self.index.insert();
        node.slot_id = slot_id;
        if slot_id == self.nodes.len() {
            self.nodes.push(node);
        } else {
            self.nodes[slot_id] = node;
        }
        slot_id
    }

    fn delete_slot(&mut self, slot_id: usize) {
        self.index.delete(slot_id);
        self.nodes[slot_id] = Node::default();
    }

    #[must_use]
    pub fn nodes_to_string_rows(&self) -> Vec<String> {
        let mut rows = Vec::new();
        let height = self.height();
        for h in (0..=height).rev() {
            for node in &self.nodes {
                if node.height == h {
                    rows.push(node.to_string());
                }
            }
        }
        rows
    }
}

impl<const D: usize> Default for Rindex<D> {
    fn default() -> Self {
        Rindex::new(10).expect("Invalid fanout")
    }
}

#[cfg(test)]
mod tests {
    use super::Rindex;
    use crate::node::Node;

    #[test]
    fn reshape() {
        let mut rindex = Rindex::default();

        // Create some point nodes
        let node_a = rindex.add_slot(Node::point([0.0, 0.0]));
        let node_b = rindex.add_slot(Node::point([0.0, 2.0]));
        let node_c = rindex.add_slot(Node::point([2.0, 0.0]));
        let node_d = rindex.add_slot(Node::point([2.0, 2.0]));

        // Create a parent node
        let mut parent = Node::default();
        parent.children = vec![node_a, node_b, node_c, node_d];
        let node_parent = rindex.add_slot(parent);

        // Reshape the parent node
        rindex.reshape(node_parent);

        // Check the parent node's sphere
        let parent = &rindex.nodes[node_parent];
        assert_eq!(parent.sphere.center, [1., 1.]);
        assert_eq!(parent.sphere.radius, 2.0_f64.sqrt());
        assert_eq!(parent.sphere.weight, 4.0);
        assert_eq!(parent.height, 1);

        // Check the parent-child relationship
        assert_eq!(rindex.nodes[node_a].parent, node_parent);
        assert_eq!(rindex.nodes[node_b].parent, node_parent);
        assert_eq!(rindex.nodes[node_c].parent, node_parent);
        assert_eq!(rindex.nodes[node_d].parent, node_parent);
    }

    #[test]
    fn split() {
        let fanout = 8;
        let mut rindex = Rindex::new(fanout).expect("Invalid fanout");

        // Create 9 point nodes, as the fanout of 8 will trigger a split
        let node_a = rindex.add_slot(Node::point([0.0, 0.0]));
        let node_b = rindex.add_slot(Node::point([0.0, 2.0]));
        let node_c = rindex.add_slot(Node::point([1.0, 1.0]));
        let node_d = rindex.add_slot(Node::point([2.0, 0.0]));
        let node_e = rindex.add_slot(Node::point([2.0, 2.0]));
        let node_w = rindex.add_slot(Node::point([10.0, 10.0]));
        let node_x = rindex.add_slot(Node::point([10.0, 12.0]));
        let node_y = rindex.add_slot(Node::point([12.0, 10.0]));
        let node_z = rindex.add_slot(Node::point([12.0, 12.0]));
        let point_nodes = vec![
            node_a, node_b, node_c, node_d, node_e, node_w, node_x, node_y, node_z,
        ];

        // Create a parent node
        let mut node = Node::default();
        node.children = point_nodes.clone();
        let node = rindex.add_slot(node);
        rindex.reshape(node);

        // Split the parent node
        let sibling = rindex.split(node);

        // Check the node's sphere
        let node = &rindex.nodes[node];
        assert_eq!(node.sphere.center, [1., 1.]);
        assert_eq!(node.sphere.radius, 2.0_f64.sqrt());
        assert_eq!(node.sphere.weight, 5.0);
        assert_eq!(node.height, 1);

        // Check the sibling's sphere
        let sibling = &rindex.nodes[sibling];
        assert_eq!(sibling.sphere.center, [11., 11.]);
        assert_eq!(sibling.sphere.radius, 2.0_f64.sqrt());
        assert_eq!(sibling.sphere.weight, 4.0);
        assert_eq!(sibling.height, 1);
    }

    #[test]
    fn update() {
        let fanout = 8;
        let mut rindex = Rindex::new(fanout).expect("Invalid fanout");
        // The tree should be empty
        assert_eq!(rindex.height(), 0);

        // Insert 8 points to fill the root node
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
}
