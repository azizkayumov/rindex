use rand::{rngs::StdRng, Rng, SeedableRng};
use rindex::{Index, LinearIndex, SSTree};

#[test]
pub fn test_random() {
    let min_pts = 3;
    let mut tree = SSTree::new(min_pts);
    let mut linear = LinearIndex::new(min_pts);

    let mut rng = StdRng::seed_from_u64(0);

    let n = 1000;
    for i in 0..n {
        let point = [rng.gen(), rng.gen()];
        let mut tree_rknns = tree.insert(point);
        let mut linear_rknns = linear.insert(point);

        tree_rknns.sort();
        linear_rknns.sort();
        assert_eq!(tree_rknns, linear_rknns);

        for j in 0..=i {
            let actual = linear.core_distance_of(j);
            let expected = tree.core_distance_of(j);
            assert_eq!(actual, expected);
        }
    }
}
