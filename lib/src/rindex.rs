pub struct Rindex<const D: usize> {
    root: usize,
}

impl<const D: usize> Rindex<D> {
    #[must_use]
    pub fn new() -> Rindex<D> {
        todo!("Implement new");
    }

    #[must_use]
    pub fn insert(&mut self, point: [f64; D]) -> usize {
        todo!("Implement insert");
    }

    pub fn delete(&mut self, point_id: usize) {
        todo!("Implement delete");
    }

    #[must_use]
    pub fn query(&self, point: [f64; D], radius: f64) -> Vec<usize> {
        todo!("Implement query");
    }

    #[must_use]
    pub fn query_neighbors(&self, point: [f64; D], k: usize) -> Vec<usize> {
        todo!("Implement query_nearest");
    }
}
