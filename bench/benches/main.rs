use criterion::{criterion_group, criterion_main};
mod uniform;

criterion_group!(
    benches,
    uniform::build,
    uniform::query,
    uniform::query_radius
);
criterion_main!(benches);
