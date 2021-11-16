use crate::model::CatZeroModel;
use crate::{AlphaMCTS, Tensor};
use mcts::transposition_table::ApproxTable;
use mcts::tree_policy::AlphaGoPolicy;
use mcts::{GameState, MCTSManager};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Player {
    Player1,
    Player2,
}

impl Player {
    pub fn other(&self) -> Player {
        match self {
            Player::Player1 => Player::Player2,
            Player::Player2 => Player::Player1,
        }
    }

    pub fn reward(&self, winner: Option<Player>) -> f32 {
        match (self, winner) {
            (Player::Player1, None) => 0.0f32,
            (Player::Player1, Some(Player::Player1)) => 1.0f32,
            (Player::Player1, Some(Player::Player2)) => -1.0f32,
            (Player::Player2, None) => todo!(),
            (Player::Player2, Some(Player::Player1)) => -1.0f32,
            (Player::Player2, Some(Player::Player2)) => 1.0f32,
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
            Player::Player2 => 1,
        }
    }
}
impl From<Player> for u8 {
    fn from(ob: Player) -> u8 {
        match ob {
            Player::Player1 => 0,
            Player::Player2 => 1,
        }
    }
}

pub trait AlphaZeroState<A>:
    Eq
    + std::hash::Hash
    + Into<Vec<Vec<Vec<u8>>>>
    + Default
    + Clone
    + GameState<Player = Player, Move = A, MoveList = Vec<A>>
where
    A: GameAction,
{
    fn is_terminal(&self) -> bool;
    fn terminal_reward(&self, searched_player: Player) -> f32;
    fn into_tensor(&self, history: Vec<&Self>) -> Vec<Vec<Vec<u8>>>;
    fn into_actions(tensor: &Tensor<Rc<RefCell<f32>>>) -> Vec<(A, Rc<RefCell<f32>>)>;
    fn get_winner(&self) -> Option<Player>;
}

pub trait GameAction: Eq + std::hash::Hash + Clone + std::fmt::Debug {}

pub trait Agent<A, S>
where
    A: GameAction,
    S: AlphaZeroState<A>,
{
    fn get_action(&mut self, state: &S) -> Option<A>;
}

pub struct AlphaAgent<'a, A, S>
where
    A: GameAction + Sync,
    S: AlphaZeroState<A> + Sync,
{
    model: &'a mut CatZeroModel<'a>,
    exploration: f64,
    playouts: usize,
    _phantom: PhantomData<(A, S)>,
}

// impl<'a, A, S> AlphaAgent<'a, A, S>
// where
//     A: GameAction + Sync,
//     S: AlphaZeroState<A> + Sync,
// {
//     pub fn new(model: &'a mut CatZeroModel<'a>, exploration: f64, playouts: usize) -> Self {
//         AlphaAgent {
//             model,
//             exploration,
//             playouts,
//             _phantom: Default::default(),
//         }
//     }

//     pub fn save(&self, path: &str) {
//         self.model.save(path).expect("Could not save model");
//     }

//     pub fn learn(&mut self, tensors: Vec<Tensor<u8>>, probs: Vec<Tensor<f32>>, rewards: Vec<f32>) {
//         self.model
//             .learn(tensors, probs, rewards, 3, 1)
//             .expect("Could not learn game!");
//     }

//     pub fn get_alpha_action(&mut self, state: S) -> Option<(A, Tensor<f32>)> {
//         let mut mcts = MCTSManager::new(
//             state,
//             AlphaMCTS::default(),
//             self.model,
//             AlphaGoPolicy::new(self.exploration),
//             ApproxTable::new(1024),
//         );

//         mcts.playout_n(self.playouts as u64);

//         let moves = *mcts
//             .principal_variation_info(1)
//             .first()
//             .expect("Could not find move");

//         let mov = moves.get_move();
//         let probs = moves
//             .child()
//             .unwrap()
//             .moves()
//             .map(|m| (*m.move_evaluation()) as f32)
//             .collect::<Vec<_>>();

//         let probs = vec![vec![probs]];

//         Some((mov.clone(), probs))
//     }
// }
// impl<'a, A, S> Agent<A, S> for AlphaAgent<'a, A, S>
// where
//     A: GameAction + Sync,
//     S: AlphaZeroState<A> + Sync,
// {
//     fn get_action(&mut self, state: &S) -> Option<A> {
//         let (action, tensor) = self.get_alpha_action(state.clone())?;

//         println!("tensor: {:?}", tensor);

//         Some(action)
//     }
// }
