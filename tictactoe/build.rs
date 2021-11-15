use std::{path::Path};

const MODEL_PATH: &'static str = "models/graph";

fn main() -> Result<(), ()> {    
    if !Path::new(MODEL_PATH).exists() {
        build_save(MODEL_PATH)?;
    }

    Ok(())
}
fn build_save(path: &str) -> Result<(), ()> {
    let mut pyenv = catzero::PyEnv::new();
    let python = pyenv.python();

    println!("Creating model");

    let nn1 = catzero::CatZeroModel::new(&python, (3, 3, 3), (1, 3, 3), 1.0, 3)
        .map_err(|_| ())?;

    println!("Create model");

    nn1.save(path).map_err(|_| ())?;
    println!("Saved model");

    Ok(())
}