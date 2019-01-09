use hashbrown::HashSet;
use mcts::MCTS;
use model::Tensor;
use model::CatZeroModel;
use std::rc::Rc;
use std::cell::RefCell;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Player {
    Player1, Player2
}

impl Player {

    pub fn other(&self) -> Player {
        match self {
            Player::Player1 => Player::Player2,
            Player::Player2 => Player::Player1
        }
    }

    pub fn reward(winner: Option<Player>, other: Self) -> f32 {
        if winner == Some(other) {
            1f32
        } else if winner.is_some() {
            -1f32
        } else {
            0f32
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


pub trait GameState<A>: Eq + std::hash::Hash + Into<Vec<Vec<Vec<u8>>>> + Default + Clone where A: GameAction {
    fn is_terminal(&self) -> bool;
    fn current_player(&self) -> Player;
    fn possible_actions(&self) -> HashSet<A>;
    fn take_action(&self, action: A) -> Self;
    fn terminal_reward(&self, searched_player: Player) -> f32;
    fn into_tensor(&self, history: Vec<&Self>) -> Vec<Vec<Vec<u8>>>;
    fn into_actions(&Tensor<Rc<RefCell<f32>>>) -> Vec<(A, Rc<RefCell<f32>>)>;
    fn get_winner(&self) -> Option<Player>;
}

pub trait GameAction: Eq + std::hash::Hash + Clone + std::fmt::Debug {

}

pub trait Agent<A, S> where A:GameAction, S:GameState<A> {
    fn get_action(&self, state: &S) -> A;
}

pub struct AlphaAgent<'a, A, S> where A:GameAction, S:GameState<A> {
    model: &'a CatZeroModel<'a>,
    searcher: MCTS<'a, S, A>
}
impl<'a, A, S> AlphaAgent<'a, A, S> where A:GameAction, S:GameState<A> {
    pub fn new(model: &'a CatZeroModel<'a>, temperature: f32) -> Self {
        AlphaAgent {
            model: model,
            searcher: MCTS::new(&model, temperature).iter_limit(Some(200))
        }
    }

    pub fn save(&self, path: &str) {
        self.model.save(path).expect("Could not save model");
    }

    pub fn learn(&self, tensors: Vec<Tensor<u8>>, probs: Vec<Tensor<f32>>, rewards: Vec<f32>) {        
        self.model.learn(tensors, probs, rewards, 3, 1).expect("Could not learn game!");
    }

    pub fn get_alpha_action(&self, state: &S) -> (A, Tensor<f32>) {
        self.searcher.alpha_search(state.clone()).clone()
    }
}
impl<'a, A, S> Agent<A, S> for AlphaAgent<'a, A, S> where A:GameAction, S:GameState<A>{
    fn get_action(&self, state: &S) -> A {
        let (action, tensor) = self.searcher.alpha_search(state.clone()).clone();

        println!("tensor: {:?}", tensor);

        action
    }
}
