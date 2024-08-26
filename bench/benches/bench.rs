use core::f64;
use std::collections::BinaryHeap;

use criterion::Criterion;
use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rindex::Rindex;

const DIMENSION: usize = 2;
const NUM_OPERATIONS: usize = 10000;
const DELETION_PROB: f64 = 0.2;
const K: usize = 100;

pub fn build(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("build");
    group.sample_size(10);

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
    group.finish();
}

pub fn query(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("query");
    group.sample_size(10);

    group.bench_function("rindex", |b| {
        b.iter(|| {
            query_rindex();
        });
    });

    group.bench_function("rstar", |b| {
        b.iter(|| {
            query_rstar();
        });
    });

    let list = build_list();
    group.bench_function("list", |b| {
        b.iter(|| {
            query_list(&list);
        });
    });
}

fn query_rindex() {
    let rindex = build_rindex();
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..NUM_OPERATIONS {
        let mut query = [0.0; DIMENSION];
        for i in 0..DIMENSION {
            query[i] = rng.gen_range(-100.0..100.0);
        }
        let (result, _) = rindex.query_neighbors(&query, K);
        assert_eq!(result.len(), K);
    }
}

fn query_rstar() {
    let rstar = build_rstar();
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..NUM_OPERATIONS {
        let mut query = [0.0; DIMENSION];
        for i in 0..DIMENSION {
            query[i] = rng.gen_range(-100.0..100.0);
        }
        let mut iter = rstar.nearest_neighbor_iter(&query);
        let results = iter.by_ref().take(K).collect::<Vec<_>>();
        assert_eq!(results.len(), K);
    }
}

fn query_list(list: &Vec<[f64; DIMENSION]>) {
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..NUM_OPERATIONS {
        let mut query = [0.0; DIMENSION];
        for i in 0..DIMENSION {
            query[i] = rng.gen_range(-100.0..100.0);
        }
        let mut results = BinaryHeap::from(vec![OrderedFloat(f64::INFINITY); K]);
        for point in list {
            let dist = query
                .iter()
                .zip(point.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist < results.peek().unwrap().0 {
                results.push(OrderedFloat(dist));
                results.pop();
            }
        }
    }
}

fn build_rindex() -> Rindex<DIMENSION> {
    let mut rng = StdRng::seed_from_u64(0);
    let mut rindex = Rindex::default();
    let mut points = Vec::new();
    for _ in 0..NUM_OPERATIONS {
        let should_delete = rng.gen_bool(DELETION_PROB);
        if should_delete && !points.is_empty() {
            let idx = rng.gen_range(0..points.len());
            let point_id = points.swap_remove(idx);
            rindex.delete(point_id);
        } else {
            let mut point = [0.0; DIMENSION];
            for i in 0..DIMENSION {
                point[i] = rng.gen_range(-100.0..100.0);
            }
            let point_id = rindex.insert(point);
            points.push(point_id);
        }
    }
    rindex
}

fn build_rstar() -> rstar::RTree<[f64; DIMENSION]> {
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
            let mut point = [0.0; DIMENSION];
            for i in 0..DIMENSION {
                point[i] = rng.gen_range(-100.0..100.0);
            }
            rstar.insert(point);
            points.push(point);
        }
    }
    rstar
}

fn build_list() -> Vec<[f64; DIMENSION]> {
    let mut rng = StdRng::seed_from_u64(0);
    let mut rindex = Rindex::default();
    let mut points = Vec::new();
    for _ in 0..NUM_OPERATIONS {
        let should_delete = rng.gen_bool(DELETION_PROB);
        if should_delete && !points.is_empty() {
            let idx = rng.gen_range(0..points.len());
            let (point_id, _) = points.swap_remove(idx);
            rindex.delete(point_id);
        } else {
            let mut point = [0.0; DIMENSION];
            for i in 0..DIMENSION {
                point[i] = rng.gen_range(-100.0..100.0);
            }
            let point_id = rindex.insert(point);
            points.push((point_id, point));
        }
    }
    points.into_iter().map(|(_, point)| point).collect()
}
