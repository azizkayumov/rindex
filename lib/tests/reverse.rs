use core::f64;
use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rindex::Rindex;
use std::collections::{BinaryHeap, HashMap};

#[test]
fn test_reverse() {
    let fanout = 10;
    let k = 5;
    let mut rindex = Rindex::new(fanout, k).expect("Invalid fanout");

    let mut rng = StdRng::seed_from_u64(0);
    let deletion_probability = 0.2;

    let num_ops = 1000;
    let mut points = Vec::new();

    // Compute the expected neighbors of each point
    let mut bruteforce = BruteForceNeighbors::new(k);

    let mut rindex_time = 0;
    let mut bruteforce_time = 0;

    for _ in 0..num_ops {
        // Randomly insert or delete a point
        let should_delete = rng.gen_bool(deletion_probability);
        if should_delete && !points.is_empty() {
            let idx = rng.gen_range(0..points.len());
            let (point_id, _) = points.swap_remove(idx);

            let start = std::time::Instant::now();
            rindex.delete(point_id);
            rindex_time += start.elapsed().as_nanos();

            let start = std::time::Instant::now();
            bruteforce.delete(point_id);
            bruteforce_time += start.elapsed().as_nanos();
        } else {
            let x = rng.gen_range(-100.0..100.0);
            let y = rng.gen_range(-100.0..100.0);

            let start = std::time::Instant::now();
            let point_id = rindex.insert([x, y]);
            rindex_time += start.elapsed().as_nanos();

            points.push((point_id, [x, y]));

            let start = std::time::Instant::now();
            bruteforce.insert(point_id, [x, y]);
            bruteforce_time += start.elapsed().as_nanos();
        }

        // Confirm the reverse neighbors query
        for (id, _) in &points {
            let (_, actual_distances) = rindex.neighbors_of(*id);
            let (_, expected_distances) = bruteforce.neighbors_of(*id);

            for (actual, expected) in actual_distances.iter().zip(expected_distances.iter()) {
                if actual.is_infinite() && expected.is_infinite() {
                    continue;
                }
                assert!(
                    actual - expected < 1e-5,
                    "Mismatch in rindex distances: {} vs {}",
                    actual,
                    expected
                );
            }
        }
    }

    println!("Rindex time:     {} ns", rindex_time);
    println!("Bruteforce time: {} ns", bruteforce_time);
}

struct BruteForceNeighbors {
    k: usize,
    points: HashMap<usize, [f64; 2]>,
    neighbors: HashMap<usize, BinaryHeap<(OrderedFloat<f64>, usize)>>,
}

impl BruteForceNeighbors {
    fn new(k: usize) -> Self {
        BruteForceNeighbors {
            k,
            points: HashMap::new(),
            neighbors: HashMap::new(),
        }
    }

    fn insert(&mut self, id: usize, point: [f64; 2]) {
        self.points.insert(id, point);
        self.update_neighbors(id);

        let rknns = self.reverse_neighbors(&point, id);
        for (distance, neighbor_id) in rknns {
            let neighbor_neighbors = self.neighbors.get_mut(&neighbor_id).unwrap();
            let neighbor_knn_dist = neighbor_neighbors.peek().unwrap().0;
            if neighbor_knn_dist > distance {
                neighbor_neighbors.pop();
                neighbor_neighbors.push((distance, id));
            }
        }
    }

    fn delete(&mut self, id: usize) {
        self.neighbors.remove(&id).unwrap();
        let point = self.points.remove(&id).unwrap();
        let rknns = self.reverse_neighbors(&point, id);
        for (_, r) in rknns {
            self.update_neighbors(r);
        }
    }

    fn update_neighbors(&mut self, point_id: usize) {
        let mut point_neighbors =
            BinaryHeap::from(vec![(OrderedFloat(f64::INFINITY), usize::MAX); self.k]);
        for (neighbor_id, neighbor) in &self.points {
            let dx = self.points[&point_id][0] - neighbor[0];
            let dy = self.points[&point_id][1] - neighbor[1];
            let distance = (dx * dx + dy * dy).sqrt();
            if distance < point_neighbors.peek().unwrap().0.into_inner() {
                point_neighbors.pop();
                point_neighbors.push((OrderedFloat(distance), *neighbor_id));
            }
        }
        self.neighbors.insert(point_id, point_neighbors);
    }

    fn reverse_neighbors(
        &mut self,
        point: &[f64; 2],
        point_id: usize,
    ) -> Vec<(OrderedFloat<f64>, usize)> {
        let mut rknns = Vec::new();
        for (neighbor_id, _) in &self.points {
            if *neighbor_id == point_id {
                continue;
            }
            let dx = point[0] - self.points[neighbor_id][0];
            let dy = point[1] - self.points[neighbor_id][1];
            let distance = (dx * dx + dy * dy).sqrt();
            let neighor_knn_dist = self
                .neighbors
                .get_mut(neighbor_id)
                .unwrap()
                .peek()
                .unwrap()
                .0
                .into_inner();
            if neighor_knn_dist >= distance {
                rknns.push((OrderedFloat(distance), *neighbor_id));
            }
        }
        rknns
    }

    fn neighbors_of(&self, id: usize) -> (Vec<usize>, Vec<f64>) {
        let neighbors = self.neighbors.get(&id).unwrap();
        let mut neighbors: Vec<(OrderedFloat<f64>, usize)> =
            neighbors.iter().map(|(dist, id)| (*dist, *id)).collect();
        neighbors.sort_by_key(|(dist, _)| OrderedFloat(*dist));
        let indices = neighbors.iter().map(|(_, id)| *id).collect();
        let distances = neighbors.iter().map(|(dist, _)| dist.0).collect();
        (indices, distances)
    }
}
