use std::marker::PhantomData;

use mctse::transposition_table::*;
use mctse::tree_policy::*;
use mctse::*;

use crate::game::GameAction;

pub struct MyEvaluator;

impl<A: GameAction + Sync, B: crate::game::AlphaZeroState<A> + Sync> Evaluator<MyMCTS<A, B>> for MyEvaluator {
    type StateEvaluation = f32;

    fn evaluate_new_state(
        &self,
        state: &B,
        moves: &MoveList<MyMCTS<A, B>>,
        handle: Option<SearchHandle<MyMCTS<A, B>>>,
    ) -> (Vec<MoveEvaluation<MyMCTS<A, B>>>, Self::StateEvaluation) {
        todo!()
    }

    fn evaluate_existing_state(
        &self,
        state: &B,
        existing_evaln: &Self::StateEvaluation,
        handle: SearchHandle<MyMCTS<A, B>>,
    ) -> Self::StateEvaluation {
        *existing_evaln
    }

    fn interpret_evaluation_for_player(
        &self,
        evaluation: &Self::StateEvaluation,
        player: &Player<MyMCTS<A, B>>,
    ) -> i64 {
        (evaluation * 100.0f32) as i64
    }
}

#[derive(Default)]
pub struct MyMCTS<A, B>(pub PhantomData<A>, pub PhantomData<B>)
where
    A: GameAction + Sync,
    B: crate::game::AlphaZeroState<A> + Sync;

impl<A: GameAction + Sync, B: crate::game::AlphaZeroState<A> + Sync> MCTS for MyMCTS<A, B> {
    type State = B;
    type Eval = MyEvaluator;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = AlphaGoPolicy;
    type TranspositionTable = ApproxTable<Self>;

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}
