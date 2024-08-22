#[derive(Clone, Copy)]
pub struct Sphere<const D: usize> {
    pub center: [f64; D],
    pub radius: f64,
}

impl<const D: usize> Sphere<D> {
    pub fn new(center: [f64; D], radius: f64) -> Sphere<D> {
        Sphere { center, radius }
    }
}
