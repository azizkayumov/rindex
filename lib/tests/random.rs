use rand::{rngs::StdRng, Rng, SeedableRng};
use rindex::Rindex;

#[test]
fn test_random() {
    let mut rindex = Rindex::default();

    // We will perform some random insertions and deletions
    let num_ops = 1000;
    let deletion_probability = 0.2; // 20% chance of deletion

    // Initialize the random number generator
    let mut rng = StdRng::seed_from_u64(0);
    let mut points = Vec::new();
    for _ in 0..num_ops {
        // Randomly insert or delete a point
        let should_delete = rng.gen_bool(deletion_probability);
        if should_delete && !points.is_empty() {
            let idx = rng.gen_range(0..points.len());
            let (point_id, _) = points.swap_remove(idx);
            rindex.delete(point_id);
        } else {
            let x = rng.gen_range(-100.0..100.0);
            let y = rng.gen_range(-100.0..100.0);
            let point_id = rindex.insert([x, y]);
            points.push((point_id, [x, y]));
        }

        // Creata a random query point and radius
        let x = rng.gen_range(-100.0..100.0);
        let y = rng.gen_range(-100.0..100.0);
        let query_point = [x, y];
        let query_radius = rng.gen_range(5.0..10.0);

        // Compute the expected results
        let mut expected = Vec::new();
        for (id, point) in &points {
            let dx = point[0] - query_point[0];
            let dy = point[1] - query_point[1];
            let distance = (dx * dx + dy * dy).sqrt();
            if distance <= query_radius {
                expected.push(*id);
            }
        }
        expected.sort();

        // Compute the actual results using the range query
        let (mut actual, _) = rindex.query(&query_point, query_radius);
        actual.sort();
        assert_eq!(expected, actual);

        // Compute the actual results using the k nearest neighbors query
        let (mut actual, _) = rindex.query_neighbors(&query_point, expected.len());
        actual.sort();
        assert_eq!(expected, actual);
    }
}
