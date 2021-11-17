use chess::{BitBoard, Board, BoardStatus, ChessMove, Color, EMPTY, Game, MoveGen};
use mcts::GameState;
use tensorflow::Tensor;


#[derive(Debug, Default, Clone, Hash)]
pub struct BoardState(Board);

impl GameState for BoardState {
    type Move = ChessMove;
    type Player = Color;
    type MoveList = Vec<ChessMove>;

    fn current_player(&self) -> Self::Player {
        self.0.side_to_move()
    }

    fn available_moves(&self) -> Self::MoveList {        
        if self.0.status() == BoardStatus::Ongoing {
            MoveGen::new_legal(&self.0).collect()
        } else {
            Vec::new()
        }
    }

    fn make_move(&mut self, mov: &Self::Move) {
        self.0.make_move(*mov, &mut self.0)
    }
}

impl BoardState {
    pub fn is_terminal(&self) -> bool {
        match self.0.status() {
            BoardStatus::Ongoing => false,
            BoardStatus::Stalemate => true,
            BoardStatus::Checkmate => true,
        }
    }

    pub fn get_winner(&self) -> Option<Color> {
        if self.available_moves().is_empty() {
            let x  = self.0.checkers();

            if x == &EMPTY {
                return None;
            } else {
                unimplemented!()
            }
        } else {
            return None;
        }
        
    }
}

impl Into<Tensor<f32>> for BoardState {
    fn into(self) -> Tensor<f32> {
        todo!()
    }
}