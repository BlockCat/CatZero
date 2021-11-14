
// use crate::model::{CatZeroModel, Tensor};
// use crate::game::{GameAction, AlphaZeroState, Player};

// use std::time::{Duration, SystemTime};
// use hashbrown::HashMap;
// use std::rc::Rc;
// use std::cell::RefCell;

// pub struct MCTS<'a, A, B> where A: AlphaZeroState<B>, B: GameAction {
//     model: &'a CatZeroModel<'a>,
//     time_limit: Option<u32>,
//     iter_limit: Option<u32>,
//     temperature: f32,
//     phantom: std::marker::PhantomData<(A, B)>
// }

// impl<'a, A, B> MCTS<'a, A, B> where A: AlphaZeroState<B>, B: GameAction  {
//     pub fn new(model: &'a CatZeroModel<'a>, temperature: f32) -> Self {
//         MCTS {
//             model,
//             time_limit: None,
//             iter_limit: None,
//             temperature: temperature,
//             phantom: std::marker::PhantomData
//         }
//     }

//     pub fn time_limit(mut self, milli_seconds: Option<u32>) -> Self {
//         self.time_limit = milli_seconds;
//         self
//     }

//     pub fn iter_limit(mut self, iter_limit: Option<u32>) -> Self {
//         self.iter_limit = iter_limit;
//         self
//     }

//     fn search_helper(&'a self, root_state: A) -> (B, MCTree<A, B>) {
//         match (self.iter_limit, self.time_limit) {
//             (None, None) => panic!("There is no search limit specified, either use a time_limit or iter_limit."),
//             (Some(_), Some(_)) => panic!("Too many limits are specified, only one allowed"),
//             _ => {}
//         }

//         let current_player = root_state.current_player();

//         let mut root_tree: MCTree<A, B> = MCTree::new(root_state, &self.model);

//         println!("Starting a search for player: {:?}", root_tree.search_player);

//         if let Some(limit) = self.iter_limit {
//             for _ in 0..limit {
//                 root_tree.execute_round();
//             }
//         }
//         if let Some(limit) = self.time_limit {
//             let expiry = SystemTime::now() + Duration::from_millis(limit as u64);
//             while SystemTime::now() < expiry {
//                 root_tree.execute_round();
//             }
//         }

//         let action = root_tree.nodes[0].actions.iter()
//             .filter(|c| c.1.child_id.is_some())
//             .max_by_key(|(a, ma)| {
//                 let qsa = ma.action_value / ma.action_count as f32;

//                 println!("({:?}) -> action: {}, count: {}, winning?: {}, preprob: {}, U: {}", *a, ma.action_value, ma.action_count, root_tree.nodes[0].win_factor, *ma.probability.borrow(), qsa);

//                 ma.action_count
//             }).unwrap().0.clone();

//         //.best_child(0, 0.0).0.clone();

//         (action, root_tree)
//     }
//     pub fn alpha_search(&'a self, root_state: A) -> (B, Tensor<f32>) {
//         let (action, root) = self.search_helper(root_state);
//         let root = &root.nodes[0];

//         let temp_probs: Vec<f32> = root.actions.iter()
//         .map(|(_, mca)| (mca.action_count as f32).powf(self.temperature))
//         .collect();

//         let normalize_constant = temp_probs.iter().sum::<f32>();

//         for (mcaction, new_prob) in root.actions.values().zip(temp_probs.iter()) {
//             *mcaction.probability.borrow_mut() = new_prob / normalize_constant;
//         }

//         let new_probs= root.raw.iter().map(|channels| {
//             channels.into_iter().map(|ydim| {
//                 ydim.into_iter().map(|cell| {
//                     cell.borrow().clone()
//                 }).collect()
//             }).collect()
//         }).collect();

//         (action, new_probs)
//     }

//     pub fn search(&'a self, root_state: A) -> B {
//         self.search_helper(root_state).0
//     }
// }

// #[derive(PartialEq, PartialOrd)]
// struct FloatWrapper(f32);

// impl Eq for FloatWrapper {}
// impl Ord for FloatWrapper {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         match self.0.partial_cmp(&other.0) {
//             None => panic!("A float could not be compared"),
//             Some(or) => or
//         }
//     }
// }

// struct MCTree<'a, A, B> where A: AlphaZeroState<B>, B: GameAction {
//     search_player: Player,
//     model: &'a CatZeroModel<'a>,
//     nodes: Vec<MCNode<A, B>>,
// }

// impl<'a, A, B> MCTree<'a, A, B> where A: AlphaZeroState<B>, B: GameAction {

//     fn new(root_state: A, model: &'a CatZeroModel<'a>) -> Self {
//         let search_player = root_state.current_player();
//         let root_node: MCNode<A, B> = MCNode::evaluate(root_state, None, model, vec!(), search_player);
//         MCTree {
//             search_player: root_node.state.current_player(),
//             model,
//             nodes: vec!(root_node),
//         }
//     }
//     fn execute_round(&mut self) {
//         // Select node
//         let selected_node = self.select_node();
//         // Get reward
//         let reward = self.nodes[selected_node].reward(self.search_player.clone());
//         // Backpropagate
//         let mut node = selected_node;

//         while let Some((parent_action, parent)) = self.nodes[node].parent.clone() {
//             self.nodes[parent].visit_count += 1;
//             let mcaction = &mut self.nodes[parent].actions.get_mut(&parent_action).unwrap();
//             mcaction.action_count += 1;
//             mcaction.action_value += reward;

//             node = parent;
//         }
//     }

//     fn select_node(&mut self) -> usize {
//         let mut node_id = 0;

//         while !self.nodes[node_id].state.is_terminal() {
//             if let Some(action) = self.find_free_action(node_id) {
//                 let expanded_id = self.expand(node_id, action);
//                 return expanded_id;
//             } else {
//                 node_id = self.best_child(node_id, 3f32).1;
//             }
//         }
//         return node_id
//     }

//     fn best_child(&self, node_id: usize, exploration_value: f32) -> (&B, usize) {
//         let node = &self.nodes[node_id];

//         let (best_action, best_child) = node.actions.iter()
//             .filter(|a| a.1.child_id.is_some())
//             .max_by_key(|(a, ma)| {
//                 let qsa = ma.action_value / ma.action_count as f32;
//                 let usa = exploration_value * *ma.probability.borrow() * (node.visit_count as f32).sqrt() / (1f32 + ma.action_count as f32);

//                 FloatWrapper(qsa + usa)
//             }).unwrap();

//         (best_action, best_child.child_id.unwrap())
//     }

//     fn find_free_action(&mut self, node_id: usize) -> Option<B> {
//         let node = &self.nodes[node_id];

//         // Find an action that has no child yet (action has not yet been done)
//         node.actions.iter()
//             .find(|(_, s)| s.child_id.is_none())
//             .map(|s| s.0.clone())
//     }

//     // TODO: Get history of previous steps,
//     // Right now, if we do a search from a state, there is no history from before that state.
//     // Eventhough the state itself has history.
//     fn history(&self, node_id: usize) -> Vec<&A> {
//         let mut history = Vec::new();
//         let mut parent = node_id; //Start at node
//         while let Some((_, e)) = &self.nodes[parent].parent {
//             parent = *e; // Go to my parent
//             history.push(parent);
//         }
//         history.into_iter().map(|id| {
//             &self.nodes[id].state
//         }).collect()
//     }

//     fn expand(&mut self, node_id: usize, action: B) -> usize {
//         let next_node_id = self.nodes.len();
//         // Set child id of node.
//         self.nodes[node_id].actions.get_mut(&action).unwrap().child_id = Some(next_node_id);

//          // Take this action and get the next state
//         let next_state = self.nodes[node_id].state.take_action(action.clone());
//         let next_node = MCNode::evaluate(next_state, Some((action, node_id)), &self.model, self.history(node_id), self.search_player.clone());
//         self.nodes.push(next_node);
//         next_node_id
//     }
// }

// struct MCAction{child_id: Option<usize>, action_count: u32, action_value: f32, probability: Rc<RefCell<f32>>}
// struct MCNode<A, B> where A: AlphaZeroState<B>, B: GameAction {
//     state: A,
//     parent: Option<(B, usize)>,
//     win_factor: f32,
//     visit_count: u32,
//     actions: HashMap<B, MCAction>,
//     raw: Tensor<Rc<RefCell<f32>>>
// }

// impl<A, B> MCNode<A, B> where A: AlphaZeroState<B>, B: GameAction {
//     fn reward(&self, search_player: Player) -> f32 {
//         if self.state.is_terminal() {
//             self.state.terminal_reward(search_player)
//         } else {
//             self.win_factor
//         }
//     }
// }

// impl<A, B> MCNode<A, B> where A: AlphaZeroState<B>, B: GameAction {

//     fn evaluate(state: A, parent: Option<(B, usize)>, model: &CatZeroModel, history: Vec<&A>, search_player: Player) -> Self {

//         let (win_factor, action_probs_raw) = model.evaluate(state.into_tensor(history))
//             .expect("Could not evaluate tensor");

//         let win_factor = if state.is_terminal() {
//             state.terminal_reward(search_player)
//         } else {
//             win_factor
//         };

//         let action_probs_raw = action_probs_raw.into_iter().map(|channels| {
//             channels.into_iter().map(|xdim| {
//                 xdim.into_iter().map(|ydim| {
//                     Rc::new(RefCell::new(ydim))
//                 }).collect()
//             }).collect()
//         }).collect();

//         let possible_actions = state.available_moves();
//         let action_probs: HashMap<_, _> = A::into_actions(&action_probs_raw).into_iter()
//             .filter(|(action, _)| possible_actions.contains(action) ) // Keep only possible probabilities
//             .map(|(action, prob)|
//                 (action, MCAction {
//                     child_id: None,
//                     action_count: 0,
//                     action_value: win_factor,
//                     probability: prob
//                 })).collect();

//         MCNode {
//             state, parent, win_factor,
//             visit_count: 0,
//             actions: action_probs,
//             raw: action_probs_raw
//         }
//     }
// }
