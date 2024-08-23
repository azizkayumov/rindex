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
        let point_a = [0.0, 0.0];
        let point_b = [0.0, 2.0];
        let point_c = [2.0, 0.0];
        let point_d = [2.0, 2.0];
        let node_a = rindex.add_node(Node::point(point_a));
        let node_b = rindex.add_node(Node::point(point_b));
        let node_c = rindex.add_node(Node::point(point_c));
        let node_d = rindex.add_node(Node::point(point_d));

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
}
