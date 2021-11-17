use ::chess::Color;
use catzero::{Player, Tensor};
use chess_state::BoardState;
use mcts::Moves;
use tmcts::ChessMCTS;

mod chess_state;
mod evaluator;
mod tmcts;

pub fn moves_to_tensor(moves: Moves<ChessMCTS>) -> Tensor<f32> {
    let mut board = vec![vec![0.0; 3]; 3];

    moves.for_each(|info| {
        let action = info.get_move();
        board[action.x][action.y] = *info.move_evaluation() as f32;
    });

    vec![board]
}

struct GameResult {
    histories: Vec<(BoardState, Tensor<f32>)>,
    winner: Option<Color>,
}

impl GameResult {
    pub fn new(winner: Option<Color>, histories: Vec<(BoardState, Tensor<f32>)>) -> GameResult {
        Self { histories, winner }
    }
}
