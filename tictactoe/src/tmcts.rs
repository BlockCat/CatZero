use crate::evaluator::MyEvaluator;
use crate::tictactoe::TicTacToeState;
use catzero::Player;
use mcts::{transposition_table::ApproxTable, tree_policy::AlphaGoPolicy, CycleBehaviour, MCTS};
use std::marker::PhantomData;

#[derive(Default, Clone)]
pub struct TicTacToeMCTS<'a>(PhantomData<&'a Player>);

impl<'a> MCTS for TicTacToeMCTS<'a> {
    type State = TicTacToeState;
    type Eval = MyEvaluator<'a>;
    type TreePolicy = AlphaGoPolicy;
    type NodeData = ();
    type TranspositionTable = ApproxTable<Self>;
    type ExtraThreadData = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::PanicWhenCycleDetected
    }
}
