use crate::{tictactoe::TicTacToeState, tmcts::TicTacToeMCTS};
use catzero::{CatZeroModel, Player};
use mcts::{Evaluator, MoveEvaluation, MoveList, SearchHandle};

#[derive(Debug, Clone)]
pub enum StateEval {
    Winner(Player),
    Draw,
    Evaluation(f32),
}
pub struct MyEvaluator<'a> {
    pub winner: Player,
    pub model: &'a CatZeroModel<'a>,
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
            let (eval, tensor) = self
                .model
                .evaluate(state.into())
                .expect("Help model not working");
            let r = StateEval::Evaluation(eval);

            let board_evaluations: Vec<f64> = moves
                .iter()
                .map(|mov| tensor[0][mov.x][mov.y] as f64)
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
    ) -> i64 {
        match evaln {
            StateEval::Winner(winner) if winner == player => 100,
            StateEval::Winner(_) => -100 as i64,
            StateEval::Draw => 0,
            StateEval::Evaluation(v) => (v * 100.0) as i64,
        }
    }
}
