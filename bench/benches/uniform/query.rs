use crate::uniform::build::{build_rindex, build_rstar};
use crate::uniform::{DIMENSION as D, K};
use core::f64;
use criterion::Criterion;
use ordered_float::OrderedFloat;
use rindex::Rindex;
use rstar::RTree;
use std::collections::BinaryHeap;

pub fn benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("query");
    //group.sample_size(10);

    let (rindex, pts) = build_rindex();
    group.bench_function("rindex", |b| {
        b.iter(|| {
            query_rindex(&rindex, &pts);
        });
    });

    let (rstar, pts) = build_rstar();
    group.bench_function("rstar", |b| {
        b.iter(|| {
            query_rstar(&rstar, &pts);
        });
    });

    group.bench_function("list", |b| {
        b.iter(|| {
            query_list(&pts);
        });
    });
}

fn query_rindex(rindex: &Rindex<D>, points: &Vec<(usize, [f64; D])>) {
    for (query_id, _) in points {
        let (neighbors, _) = rindex.neighbors_of(*query_id);
        assert_eq!(neighbors.len(), 10);
    }
}

fn query_rstar(rstar: &RTree<[f64; D]>, points: &Vec<[f64; D]>) {
    for query in points {
        let mut iter = rstar.nearest_neighbor_iter(&query);
        let results = iter.by_ref().take(K).collect::<Vec<_>>();
        assert_eq!(results.len(), K);
    }
}

fn query_list(points: &Vec<[f64; D]>) {
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
