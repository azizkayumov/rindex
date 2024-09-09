[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/azizkayumov/rindex/ci.yml?style=plastic)](#)
[![crates.io](https://img.shields.io/crates/v/rindex)](https://crates.io/crates/rindex)

# rindex
Rindex: dynamic spatial index for efficiently maintaining *k* nearest neighbors graph of multi-dimensional clustered datasets.

## Usage

The following example shows how to maintain *k* nearest neighbors using Rindex:
```rust
use rindex::Rindex;

fn main() {
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
```
Both insertion and deletion operations dynamically updates the *k* nearest neighbors for all remaining points efficiently (see the references below).

<details>
<summary>Update operations</summary>

The insertion algorithm returns an id of the newly-inserted point, store it for later usage, e.g. to delete the point:

```rust
use rindex::Rindex;

fn main() {
    let mut rindex = Rindex::default();
    let a = rindex.insert([1.0, 1.0]);
    assert_eq!(rindex.num_points(), 1);
    rindex.delete(a);
    assert_eq!(rindex.num_points(), 0);
}
```
</details>

<details>
<summary>Nearest neighbor queries</summary>
    
The traditional query operations are supported in addition to the reverse nearest neighbors query:
    
```rust
use rindex::Rindex;

fn main() {
    let k = 3;
    let mut rindex = Rindex::new(k);
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
```

</details>

## References
Rindex combines the algorithms presented in the following papers:

[1] Beckmann, N., Kriegel, H.P., Schneider, R. and Seeger, B., 1990, May. The R*-tree: An efficient and robust access method for points and rectangles. In Proceedings of the 1990 ACM SIGMOD international conference on Management of data (pp. 322-331).

[2] White, D.A. and Jain, R., 1996, February. Similarity indexing with the SS-tree. In Proceedings of the Twelfth International Conference on Data Engineering (pp. 516-523). IEEE.

[3] Yang, C. and Lin, K.I., 2001, April. An index structure for efficient reverse nearest neighbor queries. In Proceedings 17th International Conference on Data Engineering (pp. 485-492). IEEE.


## License
This project is licensed under the [Apache License, Version 2.0](LICENSE.md) - See the [LICENSE.md](https://github.com/azizkayumov/rindex/blob/main/LICENSE) file for details.