use chess::{Board, BoardStatus, ChessMove, Color, MoveGen};
use mcts::GameState;
use tensorflow::Tensor;

use crate::ChessPlayer;

#[derive(Debug, Default, Clone, Hash)]
pub struct BoardState(Board);

impl GameState for BoardState {
    type Move = ChessMove;
    type Player = ChessPlayer;
    type MoveList = Vec<ChessMove>;

    fn current_player(&self) -> Self::Player {
        ChessPlayer(self.0.side_to_move())
    }

    fn available_moves(&self) -> Self::MoveList {
        if self.0.status() == BoardStatus::Ongoing {
            MoveGen::new_legal(&self.0).collect()
        } else {
            Vec::new()
        }
    }

    fn make_move(&mut self, mov: &Self::Move) {
        self.0 = self.0.make_move_new(*mov)
    }

    fn is_terminal(&self) -> bool {
        match self.0.status() {
            BoardStatus::Ongoing => false,
            BoardStatus::Stalemate => true,
            BoardStatus::Checkmate => true,
        }
    }

    fn get_winner(&self) -> Option<Self::Player> {
        match self.0.status() {
            BoardStatus::Ongoing => unreachable!(),
            BoardStatus::Stalemate => None,
            BoardStatus::Checkmate => match self.current_player().0 {
                Color::White => Some(ChessPlayer(Color::Black)),
                Color::Black => Some(ChessPlayer(Color::White)),
            },
        }
    }
}

impl Into<Tensor<f32>> for BoardState {
    fn into(self) -> Tensor<f32> {
        todo!()
    }
}
