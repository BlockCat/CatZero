use crate::{tictactoe::TicTacToeState, tmcts::TicTacToeMCTS};
use catzero::{CatZeroModel, Player, TFModel};
use mcts::{Evaluator, MoveEvaluation, MoveList, SearchHandle};

#[derive(Debug, Clone)]
pub enum StateEval {
    Winner(Player),
    Draw,
    Evaluation(f32),
}
pub struct MyEvaluator<'a> {
    pub winner: Player,
    pub model: &'a TFModel,
}

impl<'a> Evaluator<TicTacToeMCTS<'a>> for MyEvaluator<'a> {
    type StateEvaluation = StateEval;

    fn evaluate_new_state(
        &self,
        state: &TicTacToeState,
        moves: &MoveList<TicTacToeMCTS>,
        _: Option<SearchHandle<TicTacToeMCTS>>,
    ) -> (Vec<MoveEvaluation<TicTacToeMCTS>>, Self::StateEvaluation) {
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

            let value = eval.value[0];
            let policy = eval.policy;
            let r = StateEval::Evaluation(value);

            let board_evaluations: Vec<f64> = moves
            .iter()
            .map(|mov| policy[mov.x + mov.y * 3] as f64)
            .collect();

            let sum: f64 = board_evaluations.iter().sum();
            let board_evaluations: Vec<f64> =
                board_evaluations.into_iter().map(|ev| ev / sum).collect();

            (board_evaluations, r)
        }
    }

    fn evaluate_existing_state(
        &self,
        _state: &TicTacToeState,
        evaln: &StateEval,
        _handle: SearchHandle<TicTacToeMCTS>,
    ) -> StateEval {
        evaln.clone()
    }

    fn interpret_evaluation_for_player(
        &self,
        evaln: &StateEval,
        player: &mcts::Player<TicTacToeMCTS>,
    ) -> f64 {
        match evaln {
            StateEval::Winner(winner) if winner == player => 1.0,
            StateEval::Winner(_) => -1.0,
            StateEval::Draw => 0.0,
            StateEval::Evaluation(v) => *v as f64,
        }
    }
}
