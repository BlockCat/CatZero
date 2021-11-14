use mcts::{
    transposition_table::ApproxTable, tree_policy::AlphaGoPolicy, Evaluator, GameState, MCTS,
};
pub use model::CatZeroModel;
use std::{hash::Hash, marker::PhantomData};

mod alphazero;
mod model;
mod pyenv;
// mod mcts;
pub mod game;

pub use game::Player;
pub use pyenv::PyEnv;
pub use alphazero::AlphaZero;
pub use model::Tensor;

#[derive(Debug, Default)]
pub struct AlphaMCTS<'a, S>(PhantomData<&'a S>);

impl<'a, S> MCTS for AlphaMCTS<'a, S>
where
    S: GameState + Hash + Sync,
{
    type State = S;
    type Eval = &'a CatZeroModel<'a>;
    type TreePolicy = AlphaGoPolicy;
    type NodeData = ();
    type TranspositionTable = ApproxTable<Self>;
    type ExtraThreadData = ();
}
