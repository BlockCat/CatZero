use mcts::{
    transposition_table::ApproxTable, tree_policy::AlphaGoPolicy, Evaluator, GameState, MCTS,
};
pub use model::CatZeroModel;
use std::{hash::Hash, marker::PhantomData};

mod alphazero;
mod model;
mod pyenv;
mod tf;
// mod mcts;
pub mod game;

pub use game::Player;
pub use pyenv::PyEnv;
pub use alphazero::AlphaZero;
pub use model::Tensor;
pub use tf::TFModel;

#[derive(Debug, Default)]
pub struct AlphaMCTS<'a, S>(PhantomData<&'a S>);
