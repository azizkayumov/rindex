use rindex::Rindex;
use std::{
    collections::HashMap,
    io::{BufRead, Write},
};

fn main() {
    // Read the sparse dataset (https://www.kaggle.com/datasets/joonasyoon/clustering-exercises)
    let mut data = Vec::new();
    let file = std::fs::File::open("demo/data/sparse.csv").unwrap();
    let mut skip_header = true;
    for line in std::io::BufReader::new(file).lines() {
        if skip_header {
            skip_header = false;
            continue;
        }
        let line = line.unwrap();
        let mut iter = line.split(',');
        let x = iter.next().unwrap().parse::<f64>().unwrap();
        let y = iter.next().unwrap().parse::<f64>().unwrap();
        data.push([x, y]);
    }

    // Configure the tree: we maintain 3 nearest neighbors for each point
    let k = 10;
    let mut tree = Rindex::new(k);

    // Perform random insertions and deletions
    let mut point_ids = HashMap::new();
    for (order, point) in data.iter().enumerate() {
        let point_id = tree.insert(*point);
        point_ids.insert(point_id, order);
    }
    println!("Tree height: {}", tree.height());

    let filename = "demo/data/knn.csv";
    let mut file = std::fs::File::create(filename).unwrap();
    for (point_id, point_order) in &point_ids {
        let (neighbors, _) = tree.neighbors_of(*point_id);
        for neighbor in neighbors {
            let neighbor_order = point_ids[&neighbor];
            file.write_all(format!("{point_order},{neighbor_order}\n").as_bytes())
                .unwrap();
        }
    }
}
