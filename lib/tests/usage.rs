use rindex::Rindex;

#[allow(unused_variables)]
#[test]
fn test_main_usage() {
    let k = 3; // maintain 3 nearest neighbors for each point
    let mut rindex = Rindex::new(k);

    // Insert some points
    let a = rindex.insert([1.0, 1.0]);
    let b = rindex.insert([2.0, 2.0]);
    let c = rindex.insert([3.0, 3.0]);
    let d = rindex.insert([20.0, 20.0]);

    // Check k nearest neighbors of point a
    let (neighbors, distances) = rindex.neighbors_of(a);
    assert_eq!(neighbors, vec![a, b, c]);

    // Remove point b
    rindex.delete(b);

    // Check k nearest neighbors of point a again
    let (neighbors, distances) = rindex.neighbors_of(a);
    assert_eq!(neighbors, vec![a, c, d]); // b is not in the result
}

#[allow(unused_variables)]
#[test]
fn test_update_operations() {
    let mut rindex = Rindex::default();
    let a = rindex.insert([1.0, 1.0]);
    assert_eq!(rindex.num_points(), 1);
    rindex.delete(a);
    assert_eq!(rindex.num_points(), 0);
}

#[allow(unused_variables)]
#[test]
fn test_query_operations() {
    let k = 3;
    let mut rindex = Rindex::new(k);

    // Insert some points
    let a = rindex.insert([1.0, 1.0]);
    let b = rindex.insert([2.0, 2.0]);
    let c = rindex.insert([3.0, 3.0]);
    let d = rindex.insert([20.0, 20.0]);

    let query_point = [0.0, 0.0];

    // Range queries: find all points within query_radius distance
    let query_radius = 10.0;
    let (neighbors, distances) = rindex.query(&query_point, query_radius);
    assert_eq!(neighbors, vec![a, b, c]);

    // Nearest neighbors: find 3 nearest neighbors of the query point
    let (neighbors, distances) = rindex.query_neighbors(&query_point, 3);
    assert_eq!(neighbors, vec![a, b, c]);

    // Reverse nearest neighbors: find such points that sees the query point
    // as one of their 3 nearest neighbors
    let (neighbors, distances) = rindex.query_reverse(&[0.0, 0.0]);
    assert_eq!(neighbors, vec![a]);
}
