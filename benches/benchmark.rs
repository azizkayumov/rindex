use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rindex::Index;

fn benchmark(criterion: &mut Criterion) {
    let k = 10;
    let n = 10000;
    let seed = 0;

    let mut group = criterion.benchmark_group("Insert + RkNN");
    group.sample_size(10);

    group.bench_function("SSTree", |b| b.iter(|| bench_sstree(k, n, seed)));
    group.bench_function("Linear", |b| b.iter(|| bench_linear(k, n, seed)));
}

criterion_group!(benches, benchmark);
criterion_main!(benches);

fn bench_sstree(k: usize, n: usize, seed: u64) {
    let mut tree = rindex::SSTree::new(k);
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..n {
        let point = [rng.gen(), rng.gen()];
        tree.insert(point);
    }
}

fn bench_linear(k: usize, n: usize, seed: u64) {
    let mut tree = rindex::LinearIndex::new(k);
    let mut rng = StdRng::seed_from_u64(seed);
    for _ in 0..n {
        let point = [rng.gen(), rng.gen()];
        tree.insert(point);
    }
}
