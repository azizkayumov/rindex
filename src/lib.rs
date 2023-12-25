mod distance;
mod index;
#[allow(clippy::module_name_repetitions)]
mod linear;
mod tree;

pub use index::Index;
pub use linear::LinearIndex;
pub use tree::sstree::SSTree;
