mod tictactoe;

use hashbrown::HashSet;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Player {
    Player1, Player2
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

pub trait GameAction: Eq + std::hash::Hash + Clone {

}

pub trait Game<A, S> where A: GameAction, S:GameState<A>  {
    fn new() -> Self;    
    fn get_state(&self) -> &S;
    
}

pub trait Agent<A, S> where A:GameAction, S:GameState<A> {
    fn get_action(&self, state: &S) -> A;
}