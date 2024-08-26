use crate::distance::euclidean;

pub struct Sphere<const D: usize> {
    pub center: [f64; D],
    pub radius: f64,
    pub weight: f64,
    pub variance: [f64; D],
}

impl<const D: usize> Sphere<D> {
    pub fn new(center: [f64; D], radius: f64, weight: f64) -> Sphere<D> {
        Sphere {
            center,
            radius,
            weight,
            variance: [0.0; D],
        }
    }

    pub fn point(point: [f64; D]) -> Sphere<D> {
        Self::new(point, 0.0, 1.0)
    }

    pub fn min_distance(&self, point: &[f64; D]) -> f64 {
        let dist = euclidean(&self.center, point);
        (dist - self.radius).max(0.0)
    }

    pub fn max_distance(&self, point: &[f64; D]) -> f64 {
        euclidean(&self.center, point) + self.radius
    }
}

impl<const D: usize> Default for Sphere<D> {
    fn default() -> Self {
        Sphere {
            center: [0.0; D],
            radius: 0.0,
            weight: 0.0,
            variance: [0.0; D],
        }
    }
}
