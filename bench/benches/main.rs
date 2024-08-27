use criterion::{criterion_group, criterion_main};
mod bench;

criterion_group!(benches, bench::benchmark);
criterion_main!(benches);
