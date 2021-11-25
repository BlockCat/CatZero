use crate::{ChessPlayer, chess_state::BoardState};
use catzero::{AlphaEvaluator, AlphaGame, DefaultPlayer};
use mcts::{transposition_table::ApproxTable, tree_policy::AlphaGoPolicy, CycleBehaviour, MCTS};
use std::marker::PhantomData;

#[derive(Default, Clone)]
pub struct ChessMCTS<'a>(PhantomData<&'a ChessPlayer>);

impl<'a> AlphaGame for ChessMCTS<'a> {
    fn get_exploration(&self) -> f64 {
        3.5
    }

    fn get_playouts(&self) -> usize {
        1_000
    }

    fn moves_to_tensor(moves: &Vec<&mcts::MoveInfo<Self>>) -> catzero::Tensor<f32> {
        todo!()
    }
}

impl<'a> MCTS for ChessMCTS<'a> {
    type State = BoardState;
    type Eval = AlphaEvaluator<Self>;
    type TreePolicy = AlphaGoPolicy;
    type NodeData = ();
    type TranspositionTable = ApproxTable<Self>;
    type ExtraThreadData = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::PanicWhenCycleDetected
    }
}
