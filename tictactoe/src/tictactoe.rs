use std::fmt::Display;

use catzero::{Player, Tensor};
use mcts::GameState;

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct TicTacToeAction {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug, Default, PartialEq, Eq, std::hash::Hash, Clone)]
pub struct TicTacToeState {
    current_player: Player,
    board: [[Option<Player>; 3]; 3],
}

impl Display for TicTacToeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r1 = self.board[0]
            .iter()
            .map(|c| match c {
                Some(Player::Player1) => 'X',
                Some(Player::Player2) => 'O',
                None => ' ',
            })
            .collect::<String>();
        let r2 = self.board[1]
            .iter()
            .map(|c| match c {
                Some(Player::Player1) => 'X',
                Some(Player::Player2) => 'O',
                None => ' ',
            })
            .collect::<String>();
        let r3 = self.board[2]
            .iter()
            .map(|c| match c {
                Some(Player::Player1) => 'X',
                Some(Player::Player2) => 'O',
                None => ' ',
            })
            .collect::<String>();
        f.write_fmt(format_args!("Turn: {:?}\n", self.current_player))?;
        f.write_fmt(format_args!("{}\n", r1))?;
        f.write_fmt(format_args!("{}\n", r2))?;
        f.write_fmt(format_args!("{}\n", r3))?;

        Ok(())
    }
}

impl TicTacToeState {
    pub fn is_terminal(&self) -> bool {
        match self.get_winner() {
            Some(_) => true,
            None => self.board.iter().flatten().all(Option::is_some),
        }
    }

    pub fn get_winner(&self) -> Option<Player> {
        for line in &[
            // Rows
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            // Cols
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            // Diags
            [(0, 0), (1, 1), (2, 2)],
            [(2, 0), (1, 1), (0, 2)],
        ] {
            if line
                .into_iter()
                .all(|&(x, y)| self.board[y][x] == Some(Player::Player1))
            {
                return Some(Player::Player1);
            }
            if line
                .into_iter()
                .all(|&(x, y)| self.board[y][x] == Some(Player::Player2))
            {
                return Some(Player::Player2);
            }
        }
        None
    }
}

impl GameState for TicTacToeState {
    type Move = TicTacToeAction;
    type Player = Player;
    type MoveList = Vec<TicTacToeAction>;

    fn current_player(&self) -> Self::Player {
        self.current_player.clone()
    }

    fn available_moves(&self) -> Self::MoveList {
        if self.is_terminal() {
            Vec::new()
        } else {
            self.board
                .iter()
                .enumerate()
                .map(|(y, row)| {
                    row.iter()
                        .enumerate()
                        .filter_map(move |(x, cell)| match cell {
                            None => Some(TicTacToeAction { x: x, y: y }),
                            _ => None,
                        })
                })
                .flatten()
                .collect()
        }
    }

    fn make_move(&mut self, mov: &Self::Move) {
        self.board[mov.y][mov.x] = Some(self.current_player());
        self.current_player = self.current_player.other();
    }
}

impl Into<tensorflow::Tensor<f32>> for TicTacToeState {
    fn into(self) -> tensorflow::Tensor<f32> {
        let mut tensor = tensorflow::Tensor::new(&[1, 3, 3, 3]);

        let cross_plane = self
            .board
            .iter()
            .flatten()
            .map(|cell| (cell == &Some(Player::Player1)) as u8 as f32);

        let circle_plane = self
            .board
            .iter()
            .flatten()
            .map(|cell| (cell == &Some(Player::Player2)) as u8 as f32);

        let turn_plane = match self.current_player {
            Player::Player1 => vec![vec![0f32; 3]; 3].into_iter().flatten(),
            Player::Player2 => vec![vec![1f32; 3]; 3].into_iter().flatten(),
        };

        let l = cross_plane
            .chain(circle_plane)
            .chain(turn_plane)
            .collect::<Vec<f32>>();

        for (index, v) in l.into_iter().enumerate() {
            tensor[index] = v;
        }

        tensor
    }
}

// Create input
impl Into<Tensor<u8>> for &TicTacToeState {
    fn into(self) -> Tensor<u8> {
        let cross_plane = self
            .board
            .iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell| (cell == &Some(Player::Player1)) as u8)
                    .collect()
            })
            .collect();
        let circle_plane = self
            .board
            .iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell| (cell == &Some(Player::Player2)) as u8)
                    .collect()
            })
            .collect();

        let turn_plane = match self.current_player {
            Player::Player1 => vec![vec![0u8; 3]; 3],
            Player::Player2 => vec![vec![1u8; 3]; 3],
        };

        vec![cross_plane, circle_plane, turn_plane]
    }
}

impl Into<Tensor<u8>> for TicTacToeState {
    fn into(self) -> Tensor<u8> {
        (&self).into()
    }
}

// use catzero::game::{Agent, AlphaZeroState, GameAction, Player};
// use catzero::Tensor;
// use std::cell::RefCell;
// use std::collections::hash_map::DefaultHasher;
// use std::hash::{Hash, Hasher};
// use std::rc::Rc;
// use mcts::GameState
// impl GameTrait for TicTacToeState {
//     type Player = Option<Player>;
//     type Move = TicTacToeAction;

//     fn into_tensor(&self, history: Vec<&Self>) -> Vec<Vec<Vec<u8>>> {
//         let mut current_tensor = self.into_vec();
//         let prev_tensor = match history.last() {
//             Some(last) => last.into_vec(),
//             None => vec![vec!(vec!(0u8; 3); 3); 2],
//         };

//         let colour_index: u8 = (&self.current_player).into();
//         let colour_tensor = vec![vec!(colour_index; 3); 3];

//         current_tensor.extend(prev_tensor);
//         current_tensor.push(colour_tensor);
//         current_tensor
//     }

//     fn into_actions(probs: &Tensor<Rc<RefCell<f32>>>) -> Vec<(TicTacToeAction, Rc<RefCell<f32>>)> {
//         // TODO: Add Dirichlet noise here
//         probs[0]
//             .iter()
//             .enumerate()
//             .map(|(y, row)| {
//                 row.iter()
//                     .enumerate()
//                     .map(move |(x, prob)| (TicTacToeAction { x: x, y: y }, Rc::clone(prob)))
//             })
//             .flatten()
//             .collect()
//     }
// }

// impl Into<Vec<Vec<Vec<u8>>>> for TicTacToeState {
//     fn into(self) -> Vec<Vec<Vec<u8>>> {
//         let cross = self
//             .board
//             .iter()
//             .map(|row| {
//                 row.into_iter()
//                     .map(|cell| (cell == &Some(Player::Player1)) as u8)
//                     .collect()
//             })
//             .collect();
//         let circle = self
//             .board
//             .iter()
//             .map(|row| {
//                 row.into_iter()
//                     .map(|cell| (cell == &Some(Player::Player2)) as u8)
//                     .collect()
//             })
//             .collect();

//         vec![cross, circle]
//     }
// }
