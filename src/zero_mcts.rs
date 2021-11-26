use crate::{AlphaGame, TFModel};
use mcts::{tree_policy::TreePolicy, Evaluator, GameState};
use std::{fmt::Debug, sync::Arc};

#[derive(Debug, Clone)]
pub enum StateEval<A>
where
    A: Clone,
{
    Winner(A),
    Draw,
    Evaluation(A, f32),
}

pub struct AlphaEvaluator<A>
where
    A: AlphaGame,
{
    pub winner: mcts::Player<A>,
    pub model: Arc<TFModel>,
}

impl<A> AlphaEvaluator<A>
where
    A: AlphaGame,
{
    pub fn new(winner: mcts::Player<A>, model: Arc<TFModel>) -> Self {
        Self { winner, model }
    }
}
impl<A> Evaluator<A> for AlphaEvaluator<A>
where
    A: AlphaGame,
    A::State: Into<tensorflow::Tensor<f32>>,
    A::TreePolicy: TreePolicy<A, MoveEvaluation = f64>,
{
    type StateEvaluation = StateEval<mcts::Player<A>>;

    fn evaluate_new_state(
        &self,
        state: &A::State,
        moves: &mcts::MoveList<A>,
        _: Option<mcts::SearchHandle<A>>,
    ) -> (Vec<mcts::MoveEvaluation<A>>, Self::StateEvaluation) {
        if state.is_terminal() {
            let eval = match state.get_winner() {
                Some(winner) => StateEval::Winner(winner),
                None => StateEval::Draw,
            };
            (Vec::new(), eval)
        } else {
            let eval = self
                .model
                .evaluate(state.clone().into())
                .expect("Help model not working");

            let value = (eval.value)[0];
            let policy = eval.policy;
            let state_eval = StateEval::Evaluation(state.current_player(), value);

            let board_evaluations = A::moves_to_evaluation(moves, policy);

            let sum: f64 = board_evaluations.iter().sum();

            let board_evaluations: Vec<f64> =
                board_evaluations.into_iter().map(|ev| ev / sum).collect();

            (board_evaluations, state_eval)
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
            StateEval::Evaluation(current_player, v) if current_player == player => (*v as f64),
            StateEval::Evaluation(_, v) => -(*v as f64),
        }
    }
}
