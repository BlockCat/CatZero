use crate::evaluator::MyEvaluator;
use crate::tictactoe::TicTacToeState;
use catzero::DefaultPlayer;
use mcts::{CycleBehaviour, MCTS, transposition_table::ApproxTable, tree_policy::{AlphaGoPolicy, UCTPolicy}};
use std::marker::PhantomData;

#[derive(Default, Clone)]
pub struct TicTacToeMCTS<'a>(PhantomData<&'a DefaultPlayer>);

impl<'a> MCTS for TicTacToeMCTS<'a> {
    type State = TicTacToeState;
    type Eval = MyEvaluator<'a>;
    type TreePolicy = UCTPolicy<f64>;
    type NodeData = ();
    type TranspositionTable = ApproxTable<Self>;
    type ExtraThreadData = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::PanicWhenCycleDetected
    }
}
