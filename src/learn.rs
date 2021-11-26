// use std::{hash::Hash, marker::PhantomData, rc::Rc};

// use mcts::{
//     transposition_table::ApproxTable, tree_policy::AlphaGoPolicy, Evaluator, GameState,
//     MCTSManager, SearchTree,
// };
// use rand::{prelude::SliceRandom, Rng};

// use crate::{AlphaEvaluator, AlphaGame, TFModel, Tensor};

// pub trait Playable<A>
// where
//     A: AlphaGame,
// {
//     fn play_a_game(&self, model: Rc<TFModel>) -> GameResult<A>;
// }

// impl<A> Playable<A> for A
// where
//     A: AlphaGame<TranspositionTable = ApproxTable<Self>, Eval = AlphaEvaluator<Self>> + Clone,
//     A::State: Hash,
//     A::ExtraThreadData: Default,
// {
//     fn play_a_game(&self, model: Rc<TFModel>) -> GameResult<A> {
//         let mut rng = rand::thread_rng();
//         let mut state: A::State = Default::default();

//         let mut histories: Vec<(A::State, Tensor<f32>)> = Vec::new();

//         while !state.is_terminal() {
//             // let moves = root_node.moves().collect::<Vec<_>>();
//             let moves = find_move_information(state.clone(), self.clone(), model.clone());
//             {
//                 let v = moves.iter().map(|s| s.1).sum::<f64>() - 1.0f64;
//                 assert!(
//                     v >= -0.001 && v <= 0.001,
//                     "Move evaluation should sum up to 1, but sums up to: {}",
//                     v
//                 );
//             }

//             let weighted_action = moves                
//                 .choose_weighted(&mut rng, |i| i.1)
//                 .expect("Could not get a random action");

//             // histories.push((state.clone(), Self::moves_to_tensor(&moves)));

//             state.make_move(&weighted_action.0);
//         }

//         GameResult::new(state.get_winner(), histories)
//     }
// }

// fn find_move_information<'a, A>(
//     state: A::State,
//     manager: A,
//     model: Rc<TFModel>,
// ) -> Vec<OwnedMoveEvaluation<<A::State as GameState>::Move>>
// where
//     A: AlphaGame<TranspositionTable = ApproxTable<A>, Eval = AlphaEvaluator<A>> + Clone,
//     A::State: Hash,
//     <A::State as GameState>::Move: Clone,
//     A::ExtraThreadData: Default,
// {
//     let playouts = manager.get_playouts();
//     let tree_policy = AlphaGoPolicy::new(manager.get_exploration());
//     let eval = AlphaEvaluator::<A> {
//         winner: state.current_player(),
//         model,
//     };

//     let t = SearchTree::new(state, manager, tree_policy, eval, ApproxTable::new(1024));

//     let mut tld = Default::default();
//     for _ in 0..playouts {
//         t.playout(&mut tld);
//     }

//     let root_node = t.root_node();
//     root_node
//         .moves()
//         .map(|info| OwnedMoveEvaluation(info.get_move().clone(), *info.move_evaluation()))
//         .collect::<Vec<_>>()
// }

// struct OwnedMoveEvaluation<A>(A, f64);

// pub struct GameResult<A>
// where
//     A: AlphaGame,
// {
//     histories: Vec<(A::State, Tensor<f32>)>,
//     winner: Option<<A::State as GameState>::Player>,
// }

// impl<'a, A> GameResult<A>
// where
//     A: AlphaGame,
// {
//     pub fn new(
//         winner: Option<<A::State as GameState>::Player>,
//         histories: Vec<(A::State, Tensor<f32>)>,
//     ) -> Self {
//         Self { histories, winner }
//     }
// }
