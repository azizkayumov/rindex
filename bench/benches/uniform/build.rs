use crate::uniform::{DELETION_PROB, DIMENSION as D, K, NUM_OPERATIONS};
use core::f64;
use criterion::Criterion;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rindex::Rindex;
use rstar::RTree;

pub fn benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("build");
    //group.sample_size(10);

    group.bench_function("rindex", |b| {
        b.iter(|| {
            build_rindex();
        });
    });

    group.bench_function("rstar", |b| {
        b.iter(|| {
            build_rstar();
        });
    });
}

pub fn build_rindex() -> (Rindex<D>, Vec<(usize, [f64; D])>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut index = Rindex::new(10, K).expect("Failed to create Rindex");
    let mut points = Vec::new();
    for _ in 0..NUM_OPERATIONS {
        let should_delete = rng.gen_bool(DELETION_PROB);
        if should_delete && !points.is_empty() {
            let idx = rng.gen_range(0..points.len());
            let (point_id, _) = points.swap_remove(idx);
            index.delete(point_id);
        } else {
            let mut point = [0.0; D];
            for i in 0..D {
                point[i] = rng.gen_range(-100.0..100.0);
            }
            let point_id = index.insert(point);
            points.push((point_id, point));
        }
    }
    (index, points)
}

pub fn build_rstar() -> (RTree<[f64; D]>, Vec<[f64; D]>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut rstar = rstar::RTree::new();
    let mut points = Vec::new();
    for _ in 0..NUM_OPERATIONS {
        let should_delete = rng.gen_bool(DELETION_PROB);
        if should_delete && !points.is_empty() {
            let idx = rng.gen_range(0..points.len());
            let point = points.swap_remove(idx);
            rstar.remove(&point);
        } else {
            let mut point = [0.0; D];
            for i in 0..D {
                point[i] = rng.gen_range(-100.0..100.0);
            }
            rstar.insert(point);
            points.push(point);
        }
    }
    (rstar, points)
}
