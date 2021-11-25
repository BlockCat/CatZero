use std::fmt::Display;

use catzero::{AlphaPlayer, DefaultPlayer, Tensor};
use mcts::GameState;

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct TicTacToeAction {
    pub x: usize,
    pub y: usize,
}

#[derive(Debug, Default, PartialEq, Eq, std::hash::Hash, Clone)]
pub struct TicTacToeState {
    current_player: DefaultPlayer,
    board: [[Option<DefaultPlayer>; 3]; 3],
}

impl Display for TicTacToeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r1 = self.board[0]
            .iter()
            .map(|c| match c {
                Some(DefaultPlayer::Player1) => "X│",
                Some(DefaultPlayer::Player2) => "O│",
                None => " │",
            })
            .collect::<String>();
        let r2 = self.board[1]
            .iter()
            .map(|c| match c {
                Some(DefaultPlayer::Player1) => "X│",
                Some(DefaultPlayer::Player2) => "O│",
                None => " │",
            })
            .collect::<String>();
        let r3 = self.board[2]
            .iter()
            .map(|c| match c {
                Some(DefaultPlayer::Player1) => "X│",
                Some(DefaultPlayer::Player2) => "O│",
                None => " │",
            })
            .collect::<String>();
        f.write_fmt(format_args!("Turn: {:?}\n", self.current_player))?;
        f.write_str("┌─┬─┬─┐\n")?;
        f.write_fmt(format_args!("│{}\n", r1))?;
        f.write_str("├─┼─┼─┤\n")?;
        f.write_fmt(format_args!("│{}\n", r2))?;
        f.write_str("├─┼─┼─┤\n")?;
        f.write_fmt(format_args!("│{}\n", r3))?;
        f.write_str("└─┴─┴─┘\n")?;

        Ok(())
    }
}

impl TicTacToeState {}

impl GameState for TicTacToeState {
    type Move = TicTacToeAction;
    type Player = DefaultPlayer;
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
        self.current_player = self.current_player.next_player();
    }

    fn is_terminal(&self) -> bool {
        match self.get_winner() {
            Some(_) => true,
            None => self.board.iter().flatten().all(Option::is_some),
        }
    }

    fn get_winner(&self) -> Option<DefaultPlayer> {
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
                .all(|&(x, y)| self.board[y][x] == Some(DefaultPlayer::Player1))
            {
                return Some(DefaultPlayer::Player1);
            }
            if line
                .into_iter()
                .all(|&(x, y)| self.board[y][x] == Some(DefaultPlayer::Player2))
            {
                return Some(DefaultPlayer::Player2);
            }
        }
        None
    }
}

impl Into<tensorflow::Tensor<f32>> for TicTacToeState {
    fn into(self) -> tensorflow::Tensor<f32> {
        let mut tensor = tensorflow::Tensor::new(&[1, 2, 3, 3]);

        let player = self.current_player();
        let next_player = player.next_player();

        let cross_plane = self
            .board
            .iter()
            .flatten()
            .map(|cell| (cell == &Some(player.clone())) as u8 as f32);

        let circle_plane = self
            .board
            .iter()
            .flatten()
            .map(|cell| (cell == &Some(next_player.clone())) as u8 as f32);

        let l = cross_plane
            .chain(circle_plane)
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

        let player = self.current_player();
        let next_player = player.next_player();

        let cross_plane = self
            .board
            .iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell| (cell == &Some(player.clone())) as u8)
                    .collect()
            })
            .collect();
        let circle_plane = self
            .board
            .iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell| (cell == &Some(next_player.clone())) as u8)
                    .collect()
            })
            .collect();

        vec![cross_plane, circle_plane]
    }
}

impl Into<Tensor<u8>> for TicTacToeState {
    fn into(self) -> Tensor<u8> {
        (&self).into()
    }
}