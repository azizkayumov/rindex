use rand::{rngs::StdRng, Rng, SeedableRng};
use rindex::Rindex;
use std::io::{BufRead, Write};

fn main() {
    // Read the spreader dataset
    let mut data = Vec::new();
    let file = std::fs::File::open("demo/data/spreader2D.csv").unwrap();
    for line in std::io::BufReader::new(file).lines() {
        let line = line.unwrap();
        let mut iter = line.split(',');
        let x = iter.next().unwrap().parse::<f64>().unwrap();
        let y = iter.next().unwrap().parse::<f64>().unwrap();
        data.push([x, y]);
    }

    // Configure the tree
    let mut tree = Rindex::default();
    let mut point_ids = Vec::new();
    let deletion_probability = 0.0; // 20%

    // Perform random insertions and deletions
    let mut rng = StdRng::seed_from_u64(0);
    let mut num_insertions = 0;
    let mut num_deletions = 0;
    while !data.is_empty() {
        let random = rng.gen::<f64>();
        if random <= deletion_probability && !point_ids.is_empty() {
            let random_id = rng.gen_range(0..point_ids.len());
            let point_id = point_ids.swap_remove(random_id);
            tree.delete(point_id);
            num_deletions += 1;
        } else {
            let point = data.pop().unwrap();
            let point_id = tree.insert(point);
            point_ids.push(point_id);
            num_insertions += 1;
        }
    }

    println!("Insertions: {num_insertions}");
    println!("Deletions: {num_deletions}");
    println!("Tree height: {}", tree.height());

    let csv_rows = tree.nodes_to_string_rows();
    let filename = "demo/data/tree.csv";
    let mut file = std::fs::File::create(filename).unwrap();
    for row in csv_rows {
        file.write_all(row.as_bytes()).unwrap();
    }
}
