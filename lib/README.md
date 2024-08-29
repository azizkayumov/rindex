[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/azizkayumov/rindex/ci.yml?style=plastic)](#)
[![crates.io](https://img.shields.io/crates/v/rindex)](https://crates.io/crates/rindex)

# rindex
Rindex: dynamic spatial index for efficiently maintaining *k* nearest neighbor graphs.

## Usage

The following example shows how to maintain *k* nearest neighbors using Rindex:
```
let fanout = 10;
let k = 3;
let mut rindex = Rindex::new(fanout, k).expect("Failed to create Rindex");

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
```

Both insertion and deletion operations dynamically updates the *k* nearest neighbors for all remaining points (efficiently). 

## License
This project is licensed under the [Apache License, Version 2.0](LICENSE.md) - See the [LICENSE.md](https://github.com/azizkayumov/rindex/blob/main/LICENSE) file for details.