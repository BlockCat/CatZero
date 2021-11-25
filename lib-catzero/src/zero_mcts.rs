use std::rc::Rc;

use mcts::{Evaluator, GameState, MCTS};

use crate::{AlphaGame, TFModel};

#[derive(Clone)]
pub enum StateEval<A>
where
    A: Clone,
{
    Winner(A),
    Draw,
    Evaluation(f32),
}

pub struct AlphaEvaluator<A>
where
    A: AlphaGame,
{
    pub winner: <A::State as GameState>::Player,
    pub model: Rc<TFModel>,
}

impl<A> Evaluator<A> for AlphaEvaluator<A>
where
    A: AlphaGame,
    A::State: Into<tensorflow::Tensor<f32>>,
{
    type StateEvaluation = StateEval<<A::State as GameState>::Player>;

    fn evaluate_new_state(
        &self,
        state: &A::State,
        moves: &mcts::MoveList<A>,
        _: Option<mcts::SearchHandle<A>>,
    ) -> (Vec<mcts::MoveEvaluation<A>>, Self::StateEvaluation) {
        if state.is_terminal() {
            let eval = match state.get_winner() {
                Some(a) => StateEval::Winner(a),
                None => StateEval::Draw,
            };
            (Vec::new(), eval)
        } else {
            let eval = self
                .model
                .evaluate(state.clone().into())
                .expect("Could not evaluate");

            let value = eval.value[0];
            let policy = eval.policy;
            let state_eval = StateEval::Evaluation(value);

            let move_evaluations = unimplemented!();

            (move_evaluations, state_eval)
        }
    }

    fn evaluate_existing_state(
        &self,
        _: &A::State,
        existing_evaln: &Self::StateEvaluation,
        _: mcts::SearchHandle<A>,
    ) -> Self::StateEvaluation {
        existing_evaln.clone()
    }

    fn interpret_evaluation_for_player(
        &self,
        evaluation: &Self::StateEvaluation,
        player: &mcts::Player<A>,
    ) -> f64 {
        match evaluation {
            StateEval::Winner(winner) if winner == player => 1.0,
            StateEval::Winner(_) => -1.0,
            StateEval::Draw => 0.0,
            StateEval::Evaluation(v) => *v as f64,
        }
    }
}
