use criterion::{criterion_group, criterion_main, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rindex::Index;

const K: usize = 10;
const SEED: u64 = 0;
const N: usize = 10000;

fn benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("rknn");
    group.sample_size(10);

    group.bench_function("SSTree", |b| b.iter(|| bench_sstree()));
    group.bench_function("Linear", |b| b.iter(|| bench_linear()));
}

criterion_group!(benches, benchmark);
criterion_main!(benches);

fn bench_sstree() {
    let mut tree = rindex::SSTree::new(K);
    let pts = dataset();
    for p in pts {
        tree.rknn(p);
        tree.insert(p);
    }
}

fn bench_linear() {
    let mut linear = rindex::LinearIndex::new(K);
    let dataset = dataset();
    for point in dataset {
        linear.rknn(point);
        linear.insert(point);
    }
}

fn dataset() -> Vec<[f64; 2]> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..N).map(|_| [rng.gen(), rng.gen()]).collect()
}
