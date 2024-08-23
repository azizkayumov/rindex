use ordered_float::OrderedFloat;

use crate::{distance::euclidean, index::Index, node::Node};

pub struct Rindex<const D: usize> {
    min_fanout: usize,
    max_fanout: usize,
    reinsert_fanout: usize,
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
            root: usize::MAX,
            nodes: Vec::new(),
            index: Index::new(),
        })
    }

    #[must_use]
    pub fn insert(&mut self, _point: [f64; D]) -> usize {
        todo!("Implement insert");
    }

    pub fn delete(&mut self, _point_id: usize) {
        todo!("Implement delete");
    }

    #[must_use]
    pub fn query(&self, _point: [f64; D], _radius: f64) -> Vec<usize> {
        todo!("Implement query");
    }

    #[must_use]
    pub fn query_neighbors(&self, _point: [f64; D], _k: usize) -> Vec<usize> {
        todo!("Implement query_nearest");
    }

    fn add_node(&mut self, mut node: Node<D>) -> usize {
        let slot_id = self.index.insert();
        node.slot_id = slot_id;
        if slot_id == self.nodes.len() {
            self.nodes.push(node);
        } else {
            self.nodes[slot_id] = node;
        }
        slot_id
    }

    fn delete_node(&mut self, slot_id: usize) {
        self.index.delete(slot_id);
        self.nodes[slot_id] = Node::default();
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
        let sibling = self.add_node(Node::internal(children));
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
        let node_a = rindex.add_node(Node::point([0.0, 0.0]));
        let node_b = rindex.add_node(Node::point([0.0, 2.0]));
        let node_c = rindex.add_node(Node::point([2.0, 0.0]));
        let node_d = rindex.add_node(Node::point([2.0, 2.0]));

        // Create a parent node
        let mut parent = Node::default();
        parent.children = vec![node_a, node_b, node_c, node_d];
        let node_parent = rindex.add_node(parent);

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
        let node_a = rindex.add_node(Node::point([0.0, 0.0]));
        let node_b = rindex.add_node(Node::point([0.0, 2.0]));
        let node_c = rindex.add_node(Node::point([1.0, 1.0]));
        let node_d = rindex.add_node(Node::point([2.0, 0.0]));
        let node_e = rindex.add_node(Node::point([2.0, 2.0]));
        let node_w = rindex.add_node(Node::point([10.0, 10.0]));
        let node_x = rindex.add_node(Node::point([10.0, 12.0]));
        let node_y = rindex.add_node(Node::point([12.0, 10.0]));
        let node_z = rindex.add_node(Node::point([12.0, 12.0]));
        let point_nodes = vec![
            node_a, node_b, node_c, node_d, node_e, node_w, node_x, node_y, node_z,
        ];
        println!("Point nodes: {:?}", point_nodes);

        // Create a parent node
        let mut node = Node::default();
        node.children = point_nodes.clone();
        let node = rindex.add_node(node);
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
}
