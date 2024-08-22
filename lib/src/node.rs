use crate::sphere::Sphere;

pub struct Node<const D: usize> {
    pub slot_id: usize,
    pub height: usize,
    pub parent: usize,
    pub sphere: Sphere<D>,
    pub children: Vec<usize>,
}
