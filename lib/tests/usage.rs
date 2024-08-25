use rindex::Rindex;

#[test]
fn basic_usage() {
    let mut rindex = Rindex::default();

    // Insert some points
    let a = rindex.insert([1.0, 1.0]);
    let b = rindex.insert([2.0, 2.0]);
    let c = rindex.insert([3.0, 3.0]);
    let d = rindex.insert([20.0, 20.0]);

    // Query the tree for nearest neighbors of the query point
    let query_point = [0.0, 0.0];
    let result = rindex.query_neighbors(&query_point, 3);

    // The result should contain the points a, b, and c
    assert_eq!(result.len(), 3);
    assert!(result.contains(&a));
    assert!(result.contains(&b));
    assert!(result.contains(&c));

    // Delete the point c
    rindex.delete(c);

    // Query the tree again (c should not be in the result)
    let result = rindex.query_neighbors(&query_point, 3);
    assert_eq!(result.len(), 3);
    assert!(result.contains(&a));
    assert!(result.contains(&b));
    assert!(result.contains(&d));
}
