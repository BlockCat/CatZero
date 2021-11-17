use mcts::{
    transposition_table::ApproxTable, tree_policy::AlphaGoPolicy, Evaluator, GameState, MCTS,
};
pub use model::CatZeroModel;
use serde::{Deserialize, Serialize};
use std::{fs::File, hash::Hash, io::Write, marker::PhantomData, path::Path};

mod alphazero;
mod model;
mod pyenv;
mod tf;
// mod mcts;
pub mod game;

pub use alphazero::AlphaZero;
pub use game::Player;
pub use model::Tensor;
pub use pyenv::PyEnv;
pub use tf::TFModel;

#[derive(Debug, Default)]
pub struct AlphaMCTS<'a, S>(PhantomData<&'a S>);

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TrainingData {
    pub inputs: Vec<Tensor<u8>>,
    pub output_value: Vec<f32>,
    pub output_policy: Vec<Tensor<f32>>,
}

impl TrainingData {
    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut path_buf = std::path::PathBuf::new();
        path_buf.push(std::env::current_dir().unwrap());
        path_buf.push(path);
        let s = bincode::serialize(&self).map_err(|e| e.to_string())?;


        std::fs::create_dir_all(
            path_buf
                .as_path()
                .parent()
                .and_then(|f| f.to_str())
                .unwrap(),
        ).expect("Could not create dirs");

        let mut file = File::create(&path_buf).map_err(|e| e.to_string())?;

        file.write_all(&s).map_err(|e| e.to_string())
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let s = std::fs::read(path).map_err(|e| e.to_string())?;

        bincode::deserialize(&s).map_err(|e| e.to_string())
    }
}
