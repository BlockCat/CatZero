use std::ops::Deref;

use catzero::{AlphaPlayer, Tensor};
use chess::Color;
use chess_state::BoardState;
use mcts::Moves;
use tmcts::ChessMCTS;

mod chess_state;
mod tmcts;

// pub fn moves_to_tensor(moves: Moves<ChessMCTS>) -> Tensor<f32> {
//     let mut board = vec![vec![0.0; 3]; 3];

//     moves.for_each(|info| {
//         let action = info.get_move();
//         board[action.x][action.y] = *info.move_evaluation() as f32;
//     });

//     vec![board]
// }

#[derive(Hash, PartialEq, Eq, Clone)]
pub struct ChessPlayer(pub chess::Color);

impl Default for ChessPlayer {
    fn default() -> Self {
        Self(chess::Color::White)
    }
}

impl Deref for ChessPlayer {
    type Target = chess::Color;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AlphaPlayer for ChessPlayer {
    fn next_player(&self) -> Self {
        Self(match self.0 {
            Color::White => Color::Black,
            Color::Black => Color::White,
        })
    }

    fn reward(&self, winner: Option<Self>) -> f32 {
        todo!()
    }
}

struct GameResult {
    histories: Vec<(BoardState, Tensor<f32>)>,
    winner: Option<ChessPlayer>,
}

impl GameResult {
    pub fn new(winner: Option<ChessPlayer>, histories: Vec<(BoardState, Tensor<f32>)>) -> GameResult {
        Self { histories, winner }
    }
}
