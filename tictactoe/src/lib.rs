use catzero::{Player, Tensor};
use mcts::Moves;
use tictactoe::TicTacToeState;
use tmcts::TicTacToeMCTS;

mod evaluator;
mod tictactoe;
mod tmcts;

pub fn moves_to_tensor(moves: Moves<TicTacToeMCTS>) -> Tensor<f32> {
    let mut board = vec![vec![0.0; 3]; 3];

    moves.for_each(|info| {
        let action = info.get_move();
        board[action.x][action.y] = *info.move_evaluation() as f32;
    });

    vec![board]
}

struct GameResult {
    histories: Vec<(TicTacToeState, Tensor<f32>)>,
    winner: Option<Player>,
}

impl GameResult {
    pub fn new(
        winner: Option<Player>,
        histories: Vec<(TicTacToeState, Tensor<f32>)>,
    ) -> GameResult {
        Self { histories, winner }
    }
}
