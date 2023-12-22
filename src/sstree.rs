pub struct SSTree<const D: usize> {
    k: usize,
    branching_factor: usize,
    root: usize,
}

impl<const D: usize> SSTree<D> {
    pub fn new(k: usize) -> Self {
        SSTree {
            k,
            branching_factor: 2 * k + 1,
            root: usize::MAX,
        }
    }
}
