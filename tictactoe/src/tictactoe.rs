use catzero::{ MCTS, CatZeroModel, Tensor};
use catzero::game::{GameAction, GameState, Agent, Player};

use hashbrown::HashSet;

#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct TicTacToeState {
    current_player: Player,
    board: [[Option<Player>;3]; 3]
}

impl TicTacToeState {

    fn into_vec(&self) -> Tensor<u8> {
        
        let current_cross_board = self.board.iter().map(|row| {
            row.iter().map(|cell| (cell == &Some(Player::Player1)) as u8).collect()
        }).collect();

        let current_circle_board = self.board.iter().map(|row| {
            row.iter().map(|cell| (cell == &Some(Player::Player2)) as u8).collect()
        }).collect();

        vec!(current_cross_board, current_circle_board)
    }

    
}

impl GameState<TicTacToeAction> for TicTacToeState {
    fn is_terminal(&self) -> bool {
        match self.get_winner() {
            Some(_) => true,
            None => self.board.iter().flatten().filter(|cell| cell.is_none()).count() == 0
        }        
    }

    fn current_player(&self) -> Player {
        self.current_player.clone() 
    }

    fn possible_actions(&self) -> HashSet<TicTacToeAction> {
        self.board.iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().filter_map(move |(x, cell)| {
                 match cell {
                    None => Some(TicTacToeAction {x: x, y: y}),
                    _ => None
                }
            })
        }).flatten().collect()
    }

    fn get_winner(&self) -> Option<Player> {
        for line in &[
            // Rows
            [(0,0), (1, 0), (2,0)], [(0,1), (1, 1), (2,1)], [(0,2), (1, 2), (2,2)],
            // Cols
            [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
            // Diags
            [(0, 0), (1, 1), (2, 2)], [(2, 0), (1, 1), (0, 2)]
        ] {
            if line.into_iter().all(|&(x, y)| self.board[y][x] == Some(Player::Player1)) {
                return Some(Player::Player1);
            }
            if line.into_iter().all(|&(x, y)| self.board[y][x] == Some(Player::Player2)) {
                return Some(Player::Player2);
            }
        }
        None
    }


    fn take_action(&self, action: TicTacToeAction) -> Self {
        let mut next_state = self.clone();

        next_state.current_player = self.current_player.other();
        next_state.board[action.y][action.x] = Some(self.current_player());

        next_state
    }

    fn terminal_reward(&self, searched_player: Player) -> f32 {
        let winner = self.get_winner();

        match winner {            
            Some(ref player) if player == &searched_player => 1f32,
            Some(_) => -1f32,
            None => 0f32,            
        }
    }

    fn into_tensor(&self, history: Vec<&Self>) -> Vec<Vec<Vec<u8>>> {
        let mut current_tensor = self.into_vec();
        let prev_tensor = match history.last() {
            Some(last) => last.into_vec(),
            None => vec!(vec!(vec!(0u8; 3);3); 2)
        };

        let colour_index: u8 = (&self.current_player).into();
        let colour_tensor = vec!(vec!(colour_index; 3); 3);

        current_tensor.extend(prev_tensor);
        current_tensor.push(colour_tensor);
        current_tensor
    }

    fn into_actions(probs: Vec<Vec<Vec<f32>>>) -> Vec<(f32, TicTacToeAction)> {
        // TODO: Add Dirichlet noise here
        probs[0].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().map(move |(x, prob)| {
                (*prob, TicTacToeAction {x: x, y:y})
            })
        }).flatten().collect()
    }
}

impl Into<Vec<Vec<Vec<u8>>>> for TicTacToeState {
    fn into(self) -> Vec<Vec<Vec<u8>>> {
        let cross = self.board.iter().map(|row| row.into_iter().map(|cell| (cell == &Some(Player::Player1)) as u8).collect()).collect();
        let circle = self.board.iter().map(|row| row.into_iter().map(|cell| (cell == &Some(Player::Player2)) as u8).collect()).collect();        

        vec!(cross, circle)
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct TicTacToeAction {
    x: usize,
    y: usize,
}

impl GameAction for TicTacToeAction {}

pub struct TicTacToe<A, B> where A: Agent<TicTacToeAction, TicTacToeState>, B: Agent<TicTacToeAction, TicTacToeState> {
    player1: A,
    player2: B,
    current_state: TicTacToeState,
    do_print: bool,
}

impl<A, B> TicTacToe<A, B> where A: Agent<TicTacToeAction, TicTacToeState>, B: Agent<TicTacToeAction, TicTacToeState>  {
    
    pub fn do_print(&mut self, print: bool) {
        self.do_print = print;
    }

    fn print(&self) {
        
        let r1 = self.current_state.board[0].iter().map(|c| match c {
            Some(Player::Player1) => 'X',
            Some(Player::Player2) => 'O',
            None => ' '
        }).collect::<String>();
        let r2 = self.current_state.board[1].iter().map(|c| match c {
            Some(Player::Player1) => 'X',
            Some(Player::Player2) => 'O',
            None => ' '
        }).collect::<String>();
        let r3 = self.current_state.board[2].iter().map(|c| match c {
            Some(Player::Player1) => 'X',
            Some(Player::Player2) => 'O',
            None => ' '
        }).collect::<String>();
        println!("┌───┐");
        println!("|{}|", r1);
        println!("|{}|", r2);
        println!("|{}|", r3);
        println!("└───┘")
    }

    pub fn new(player1: A , player2: B) -> Self {
        TicTacToe {
            player1: player1,
            player2: player2,
            current_state: TicTacToeState::default(),
            do_print: false
        }
    }

    pub fn start(&mut self) -> Option<Player> {
        if self.do_print {
            println!("Starting game:");
            self.print();
        }

        while !self.current_state.is_terminal() { // When the game hasn't ended yet
            let player1_action = self.player1.get_action(&self.current_state);
            self.current_state = self.current_state.take_action(player1_action);

            if self.do_print {                
                self.print();
            }

            if self.current_state.is_terminal() {
                return self.current_state.get_winner();
            }

            let player2_action = self.player2.get_action(&self.current_state);
            self.current_state = self.current_state.take_action(player2_action);

            if self.do_print {                
                self.print();
            }
        }

        self.current_state.get_winner()
    }
}

pub struct PlayerAgent;
impl Agent<TicTacToeAction, TicTacToeState> for PlayerAgent {
    fn get_action(&self, _: &TicTacToeState) -> TicTacToeAction {
        use std::io;

        let mut buffer = String::new();
        println!("Please enter coordinations: x,y");

        io::stdin().read_line(&mut buffer).unwrap();

        let x: usize = buffer[0..1].parse().expect("X is not an integer");
        let y: usize = buffer[2..3].parse().expect("y is not an integer");

        TicTacToeAction {
            x: x,
            y: y,
        }
    }
}

