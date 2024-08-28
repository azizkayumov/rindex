mod build;
mod query;
mod query_radius;

// Benchmark parameters:
pub const DIMENSION: usize = 2;
pub const NUM_OPERATIONS: usize = 10000; // Number of operations to perform (insertions and deletions)
pub const DELETION_PROB: f64 = 0.2; // Probability of deleting a point for simulating dynamic data
pub const K: usize = 10; // Number of neighbors to query
pub const RADIUS: f64 = 5.0; // Radius for range queries
pub const RADIUS_SQUARED: f64 = RADIUS * RADIUS;

pub use build::benchmark as build;
pub use query::benchmark as query;
pub use query_radius::benchmark as query_radius;
