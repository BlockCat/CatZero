use catzero::{DefaultPlayer, Tensor};
use mcts::Moves;
use tictactoe::TicTacToeState;
use tmcts::TicTacToeMCTS;

mod evaluator;
mod tictactoe;
mod tmcts;

pub fn moves_to_tensor(moves: Moves<TicTacToeMCTS>) -> Tensor<f32> {
    let mut board = vec![vec![0.0; 3]; 3];
    let moves = moves.collect::<Vec<_>>();
    let parent_visits: u64 = moves.iter().map(|f| f.visits()).sum();    
    
    if parent_visits == 0 {
        panic!("Parent visits where 0")
    }

    let parent_visits = parent_visits as f32;
    for info in moves {
        let action = info.get_move();
        let probability = info.visits() as f32 / parent_visits;
        board[action.x][action.y] = probability;
    }
    vec![board]
}

struct GameResult {
    histories: Vec<(TicTacToeState, Tensor<f32>)>,
    winner: Option<DefaultPlayer>,
}

impl GameResult {
    pub fn new(
        winner: Option<DefaultPlayer>,
        histories: Vec<(TicTacToeState, Tensor<f32>)>,
    ) -> GameResult {
        Self { histories, winner }
    }
}
