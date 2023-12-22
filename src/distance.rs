pub fn euclidean<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y).powi(2);
    }
    sum.sqrt()
}
