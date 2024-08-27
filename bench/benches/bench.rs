use core::f64;
use std::collections::BinaryHeap;

use criterion::Criterion;
use ordered_float::OrderedFloat;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rindex::Rindex;
use rstar::RTree;

const DIMENSION: usize = 2;
const NUM_OPERATIONS: usize = 1000;
const DELETION_PROB: f64 = 0.2;
const K: usize = 10;

pub fn benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench");
    group.sample_size(10);

    group.bench_function("rindex", |b| {
        b.iter(|| {
            bench_rindex();
        });
    });

    group.bench_function("rstar", |b| {
        b.iter(|| {
            bench_rstar();
        });
    });

    group.bench_function("list", |b| {
        b.iter(|| {
            bench_list();
        });
    });

    group.finish();
}

fn bench_rindex() -> (Rindex<DIMENSION>, Vec<(usize, [f64; DIMENSION])>) {
    let mut rng = StdRng::seed_from_u64(0);
    let mut index = Rindex::new(10, K).unwrap();
    let mut points = Vec::new();
    for _ in 0..NUM_OPERATIONS {
        let should_delete = rng.gen_bool(DELETION_PROB);
        if should_delete && !points.is_empty() {
            let idx = rng.gen_range(0..points.len());
            let (point_id, _) = points.swap_remove(idx);
            index.delete(point_id);
        } else {
            let mut point = [0.0; DIMENSION];
            for i in 0..DIMENSION {
                point[i] = rng.gen_range(-100.0..100.0);
            }
            let point_id = index.insert(point);
            points.push((point_id, point));
        }
        query_rindex(&index, &points);
    }
    (index, points)
}

fn bench_rstar() {
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
        query_rstar(&rstar, &points);
    }
}

fn bench_list() -> Vec<[f64; DIMENSION]> {
    let mut rng = StdRng::seed_from_u64(0);
    let mut points = Vec::new();
    for _ in 0..NUM_OPERATIONS {
        let should_delete = rng.gen_bool(DELETION_PROB);
        if should_delete && !points.is_empty() {
            let idx = rng.gen_range(0..points.len());
            points.swap_remove(idx);
        } else {
            let mut point = [0.0; DIMENSION];
            for i in 0..DIMENSION {
                point[i] = rng.gen_range(-100.0..100.0);
            }
            points.push(point);
        }
        query_list(&points);
    }
    points
}

fn query_rindex(rindex: &Rindex<DIMENSION>, points: &Vec<(usize, [f64; DIMENSION])>) {
    for (query_id, _) in points {
        let (neighbors, _) = rindex.neighbors_of(*query_id);
        assert_eq!(neighbors.len(), K);
    }
}

fn query_rstar(rstar: &RTree<[f64; DIMENSION]>, points: &Vec<[f64; DIMENSION]>) {
    for query in points {
        let mut iter = rstar.nearest_neighbor_iter(&query);
        let results = iter.by_ref().take(K).collect::<Vec<_>>();
        assert_eq!(results.len(), K.min(points.len()));
    }
}

fn query_list(points: &Vec<[f64; DIMENSION]>) {
    for query in points {
        let mut results = BinaryHeap::from(vec![OrderedFloat(f64::INFINITY); K]);
        for point in points {
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
        assert_eq!(results.len(), K);
    }
}
