use crate::{chess_state::BoardState, evaluator::MyEvaluator};
use catzero::Player;
use mcts::{transposition_table::ApproxTable, tree_policy::AlphaGoPolicy, CycleBehaviour, MCTS};
use std::marker::PhantomData;

#[derive(Default, Clone)]
pub struct ChessMCTS<'a>(PhantomData<&'a Player>);

impl<'a> MCTS for ChessMCTS<'a> {
    type State = BoardState;
    type Eval = MyEvaluator<'a>;
    type TreePolicy = AlphaGoPolicy;
    type NodeData = ();
    type TranspositionTable = ApproxTable<Self>;
    type ExtraThreadData = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::PanicWhenCycleDetected
    }
}
