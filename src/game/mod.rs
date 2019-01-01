pub mod tictactoe;

use hashbrown::HashSet;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Player {
    Player1, Player2
}

impl Player {

    fn other(&self) -> Player {
        match self {
            Player::Player1 => Player::Player2,
            Player::Player2 => Player::Player1
        }
    }
}

impl Default for Player {
    fn default() -> Self {
        Player::Player1
    }
}

impl From<&Player> for u8 {
    fn from(ob: &Player) -> u8 {
        match ob {
            Player::Player1 => 0,
            Player::Player2 => 1
        }
    }
}
impl From<Player> for u8 {
    fn from(ob: Player) -> u8 {
        match ob {
            Player::Player1 => 0,
            Player::Player2 => 1
        }
    }
}


pub trait GameState<A>: Eq + std::hash::Hash + Into<Vec<Vec<Vec<u8>>>> where A: GameAction {
    fn is_terminal(&self) -> bool;
    fn current_player(&self) -> Player;
    fn possible_actions(&self) -> HashSet<A>;
    fn take_action(&self, action: A) -> Self;
    fn terminal_reward(&self) -> f32;
    fn into_tensor(&self, history: Vec<&Self>) -> Vec<Vec<Vec<u8>>>;
    fn into_actions(Vec<Vec<Vec<f32>>>) -> Vec<(f32, A)>;
}

pub trait GameAction: Eq + std::hash::Hash + Clone + std::fmt::Debug {

}

pub trait Game<A, S, C, D> where A: GameAction, S:GameState<A>, C: Agent<A, S>, D: Agent<A, S>  {
    fn new(player1: C, player2: D) -> Self;    
    fn start(&self);    
}

pub trait Agent<A, S> where A:GameAction, S:GameState<A> {
    fn get_action(&self, state: &S) -> A;
}