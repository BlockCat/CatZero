use mcts::{MCTSManager, MCTS};
pub use model::CatZeroModel;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Write, ops::Range, sync::Arc};

// pub mod learn;
mod model;
mod player;
mod pyenv;
mod tf;
mod utils;
mod zero_mcts;

// pub use learn::Playable;
pub use model::Tensor;
pub use player::{AlphaPlayer, DefaultPlayer};
pub use pyenv::PyEnv;
pub use tf::TFModel;
pub use utils::{play_a_game, GameResult};
pub use zero_mcts::AlphaEvaluator;

pub trait AlphaGame: MCTS {
    /// Get exploration constant
    fn get_exploration(&self) -> f64;
    /// Get amount of playouts needed
    fn get_playouts(&self) -> usize;

    /// Moves to evaluation, for MCTS
    fn moves_to_evaluation(
        moves: &mcts::MoveList<Self>,
        policy: tensorflow::Tensor<f32>,
    ) -> Vec<mcts::MoveEvaluation<Self>>;

    /// Moves to tensor, for training (output policy)
    fn moves_to_tensorflow(moves: Vec<&mcts::MoveInfo<Self>>) -> tensorflow::Tensor<f32>;

    fn create_manager(
        state: Self::State,
        exploration_constant: f64,
        playouts: usize,
        model: Arc<TFModel>,
    ) -> MCTSManager<Self>;
}

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
        )
        .expect("Could not create dirs");

        let mut file = File::create(&path_buf).map_err(|e| e.to_string())?;

        file.write_all(&s).map_err(|e| e.to_string())
    }

    pub fn len(&self) -> usize {
        self.inputs.len()
    }

    pub fn extend(&mut self, other: Self) {
        self.inputs.extend(other.inputs);
        self.output_policy.extend(other.output_policy);
        self.output_value.extend(other.output_value);
    }

    pub fn load(path: &str) -> Result<Self, String> {
        let s = std::fs::read(path).map_err(|e| e.to_string())?;

        bincode::deserialize(&s).map_err(|e| e.to_string())
        // unimplemented!()
    }

    pub fn print(&self, index: Range<usize>) {
        let inputs = &self.inputs[index.clone()];
        let output_p = &self.output_policy[index.clone()];
        let output_v = &self.output_value[index.clone()];

        for (input, (p, v)) in inputs.iter().zip(output_p.iter().zip(output_v)) {
            println!("INPUT: {:?}", input);
            println!("POLICY: {:?}", p);
            println!("VALUE: {}", v);
        }
    }
}
