use criterion::{criterion_group, criterion_main};
mod bench;

criterion_group!(benches, bench::build, bench::query);
criterion_main!(benches);
