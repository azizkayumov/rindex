use crate::uniform::build::{build_rindex, build_rstar};
use crate::uniform::{DIMENSION as D, RADIUS, RADIUS_SQUARED};
use core::f64;
use criterion::Criterion;
use rindex::Rindex;
use rstar::RTree;

pub fn benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("query_radius");
    //group.sample_size(10);

    let (rindex, pts) = build_rindex();
    group.bench_function("rindex", |b| {
        b.iter(|| {
            query_range_rindex(&rindex, &pts);
        });
    });

    let (rstar, pts) = build_rstar();
    group.bench_function("rstar", |b| {
        b.iter(|| {
            query_range_rstar(&rstar, &pts);
        });
    });

    group.bench_function("list", |b| {
        b.iter(|| {
            query_range_list(&pts);
        });
    });
}

fn query_range_rindex(rindex: &Rindex<D>, points: &Vec<(usize, [f64; D])>) {
    for (query_id, query) in points {
        let (neighbors, _) = rindex.query(&query, RADIUS);
        assert!(neighbors.contains(query_id));
    }
}

fn query_range_rstar(rstar: &RTree<[f64; D]>, points: &Vec<[f64; D]>) {
    for query in points {
        let result = rstar
            .locate_within_distance(*query, RADIUS_SQUARED)
            .collect::<Vec<_>>();
        assert!(result.contains(&query));
    }
}

fn query_range_list(points: &Vec<[f64; D]>) {
    for query in points {
        let mut results = Vec::new();
        for point in points {
            let dist = query
                .iter()
                .zip(point.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            if dist <= RADIUS {
                results.push(dist);
            }
        }
        assert!(results.contains(&0.0));
    }
}
