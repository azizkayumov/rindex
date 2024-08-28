[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/azizkayumov/rindex/ci.yml?style=plastic)](#)
[![crates.io](https://img.shields.io/crates/v/rindex)](https://crates.io/crates/rindex)

# rindex
Rindex: fully dynamic nearest neighbor search index for high-dimensional clustered datasets.

## Usage

The following example shows how to update and query Rindex:
```
let mut rindex = Rindex::default();

// Insert some points
let a = rindex.insert([1.0, 1.0]);
let b = rindex.insert([2.0, 2.0]);
let c = rindex.insert([3.0, 3.0]);
let d = rindex.insert([20.0, 20.0]);

// Query the tree for nearest neighbors of the query point
let query_point = [0.0, 0.0];
let (indices, _distances) = rindex.query_neighbors(&query_point, 3);

// The result should contain the points a, b, and c
assert_eq!(indices.len(), 3);
assert!(indices.contains(&a));
assert!(indices.contains(&b));
assert!(indices.contains(&c));

// Delete the point c
rindex.delete(c);

// Query the tree again (c should not be in the result)
let (indices, _distances) = rindex.query_neighbors(&query_point, 3);
assert_eq!(indices.len(), 3);
assert!(indices.contains(&a));
assert!(indices.contains(&b));
assert!(indices.contains(&d));
```

## License
This project is licensed under the [Apache License, Version 2.0](LICENSE.md) - See the [LICENSE.md](https://github.com/azizkayumov/rindex/blob/main/LICENSE) file for details.