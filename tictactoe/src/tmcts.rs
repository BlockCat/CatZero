use std::fmt::Display;
use std::marker::PhantomData;

use catzero::Player;
use catzero::CatZeroModel;
use mcts::transposition_table::*;
use mcts::tree_policy::*;
use mcts::*;

use crate::evaluator::MyEvaluator;
use crate::tictactoe::TicTacToeState;

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
