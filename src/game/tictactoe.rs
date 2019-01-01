use crate::game::{Game, GameAction, GameState, Agent, Player};
use crate::mcts::MCTS;
use crate::model::CatZeroModel;
use hashbrown::HashSet;

#[derive(Default, PartialEq, Eq, Hash, Clone)]
pub struct TicTacToeState {
    current_player: Player,
    board: [[Option<Player>;3]; 3]
}

impl TicTacToeState {
    fn into_vec(&self) -> Vec<Vec<Vec<u8>>> {
        let current_cross_board: Vec<Vec<u8>> = self.board.iter().map(|row| {
            row.iter().map(|cell| {
                match cell {
                    Some(Player::Player1) => 1,
                    _ => 0
                }
            }).collect()
        }).collect();
        let current_circle_board: Vec<Vec<u8>> = self.board.iter().map(|row| {
            row.iter().map(|cell| {
                match cell {
                    Some(Player::Player2) => 1,
                    _ => 0
                }
            }).collect()
        }).collect();
        vec!(current_cross_board, current_circle_board)
    }
}

impl GameState<TicTacToeAction> for TicTacToeState {
    fn is_terminal(&self) -> bool {
        self.board.iter().flatten().filter(|cell| match cell {
            Some(_) => true,
            None => false
        }).count() == 0
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


    fn take_action(&self, action: TicTacToeAction) -> Self {
        let mut next_state = self.clone();
        next_state.current_player = self.current_player.other();
        next_state.board[action.y][action.x] = Some(self.current_player());

        next_state
    }

    fn terminal_reward(&self) -> f32 {
        0f32
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
        probs[0].iter().enumerate().map(|(y, row)| {
            row.iter().enumerate().map(move |(x, prob)| {
                (*prob, TicTacToeAction {x: x, y:y})
            })
        }).flatten().collect()
    }
}

impl Into<Vec<Vec<Vec<u8>>>> for TicTacToeState {
    fn into(self) -> Vec<Vec<Vec<u8>>> {
        let cross = self.board.iter()
            .map(|row| row.into_iter().map(|cell| match cell {
                Some(Player::Player1) => 1,
                Some(Player::Player2) | None => 0                
            }).collect()).collect();

        let circle = self.board.iter()
            .map(|row| row.into_iter().map(|cell| match cell {
                Some(Player::Player2) => 1,
                Some(Player::Player1) | None => 0                
            }).collect()).collect();

        vec!(cross, circle)
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct TicTacToeAction {
    x: usize,
    y: usize,
}

impl GameAction for TicTacToeAction {

}

pub struct TicTacToe<A, B> where A: Agent<TicTacToeAction, TicTacToeState>, B: Agent<TicTacToeAction, TicTacToeState> {
    player1: A,
    player2: B,
}

impl<A, B> Game<TicTacToeAction, TicTacToeState, A, B> for TicTacToe<A, B> where A: Agent<TicTacToeAction, TicTacToeState>, B: Agent<TicTacToeAction, TicTacToeState>  {
    fn new(player1: A, player2: B) -> Self {
        TicTacToe { player1, player2 }
    }

    fn start(&self) {

    }
}

pub struct PlayerAgent;
impl Agent <TicTacToeAction, TicTacToeState> for PlayerAgent {
    fn get_action(&self, state: &TicTacToeState) -> TicTacToeAction {
        panic!()
    }
}

pub struct AlphaAgent<'a> {
    searcher: MCTS<'a, TicTacToeState, TicTacToeAction>
}
impl<'a> AlphaAgent<'a> {
    fn new(model: &'a CatZeroModel<'a>) -> Self {
        AlphaAgent {
            searcher: MCTS::new(&model)
        }
    }
}
impl<'a> Agent <TicTacToeAction, TicTacToeState> for AlphaAgent<'a> {
    fn get_action(&self, state: &TicTacToeState) -> TicTacToeAction {
        self.searcher.search(state.clone()).clone()
    }
}
