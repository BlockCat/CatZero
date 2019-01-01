use crate::game::{Game, GameAction, GameState, Agent, Player};
use crate::mcts::MCTS;
use crate::model::CatZeroModel;
use hashbrown::HashSet;

#[derive(Default, PartialEq, Eq, Hash, Clone)]
struct TicTacToeState {
    board: [[Option<Player>;3]; 3]
}

impl GameState<TicTacToeAction> for TicTacToeState {
    fn is_terminal(&self) -> bool {
        false
    }

    fn current_player(&self) -> Player {
        panic!()
    }

    fn possible_actions(&self) -> HashSet<TicTacToeAction> {
        panic!()
    }


    fn take_action(&self, action: TicTacToeAction) -> Self {
        panic!()
    }

    fn terminal_reward(&self) -> f32 {
        0f32
    }


    fn into_tensor(&self, history: Vec<&Self>) -> Vec<Vec<Vec<u8>>> {
        panic!()
    }

    fn into_actions(probs: Vec<Vec<Vec<f32>>>) -> Vec<(f32, TicTacToeAction)> {
        panic!()
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

#[derive(Hash, Eq, PartialEq, Clone)]
struct TicTacToeAction {
    x: usize,
    y: usize,
}

impl GameAction for TicTacToeAction {

}

pub struct TicTacToe {
    state: TicTacToeState
}

impl Game<TicTacToeAction, TicTacToeState> for TicTacToe {
    fn new() -> TicTacToe {
        TicTacToe {
            state: TicTacToeState::default()
        }
    }

    fn get_state(&self) -> &TicTacToeState {
        &self.state
    }

    
}

struct PlayerAgent;
impl Agent <TicTacToeAction, TicTacToeState> for PlayerAgent {
    fn get_action(&self, state: &TicTacToeState) -> TicTacToeAction {
        panic!()
    }
}

struct AlphaAgent<'a> {
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
