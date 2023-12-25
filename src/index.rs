pub trait Index<const D: usize> {
    fn insert(&mut self, point: [f64; D]) -> Vec<usize>;
    fn query_range(&self, point_index: usize, range: f64) -> Vec<usize>;

    fn core_distance_of(&self, point_index: usize) -> f64;
    fn neighbors_of(&self, point_index: usize) -> Vec<usize>;
    fn num_points(&self) -> usize;
}
