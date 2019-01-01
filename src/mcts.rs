use crate::model::CatZeroModel;
use crate::game::{GameAction, GameState};
use std::time::{Duration, SystemTime};
use hashbrown::HashMap;
use hashbrown::HashSet;

pub struct MCTS<'a, A, B> where A: GameState<B>, B: GameAction {
    model: &'a CatZeroModel<'a>,
    time_limit: Option<u32>,
    iter_limit: Option<u32>,
    phantom: std::marker::PhantomData<A>,
    phantom_2: std::marker::PhantomData<B>,
}

impl<'a, A, B> MCTS<'a, A, B> where A: GameState<B>, B: GameAction  {
    pub fn new(model: &'a CatZeroModel<'a>) -> Self {
        panic!()
    }

    pub fn time_limit(&mut self, milli_seconds: Option<u32>) -> &mut Self {
        self.time_limit = milli_seconds;
        self
    }

    pub fn iter_limit(&mut self, iter_limit: Option<u32>) -> &mut Self {
        self.iter_limit = iter_limit;
        self
    }
    
    pub fn search(&'a self, root_state: A) -> B {
        match (self.iter_limit, self.time_limit) {
            (None, None) => panic!("There is no search limit specified, either use a time_limit or iter_limit."),
            (Some(_), Some(_)) => panic!("Too many limits are specified, only one allowed"),
            _ => {}
        }

        let mut root_tree: MCTree<A, B> = MCTree::new(root_state, &self.model);

        if let Some(limit) = self.iter_limit {
            for _ in 0..limit {
                root_tree.execute_round();
            }
        }
        if let Some(limit) = self.time_limit {
            let expiry = SystemTime::now() + Duration::from_millis(limit as u64);
            while SystemTime::now() < expiry {
                root_tree.execute_round();
            }
        }
        
        root_tree.best_child(0, 0f32).0.clone()
    }
}

struct MCTree<'a, A, B> where A: GameState<B>, B: GameAction {
    model: &'a CatZeroModel<'a>,
    nodes: Vec<MCNode<A, B>>,
    //states: HashSet<A, u32>
}

impl<'a, A, B> MCTree<'a, A, B> where A: GameState<B>, B: GameAction {

    fn new(root_state: A, model: &'a CatZeroModel<'a>) -> Self {
        let root_node = MCNode::evaluate(root_state, None, model, vec!());        
        MCTree {
            model,
            nodes: vec!(root_node),
      //      states: HashSet::new()
        }
    }
    fn execute_round(&mut self) {
        // Select node
        let selected_node = self.select_node();
        // Get reward
        let reward = selected_node.reward();
        // Backpropagate

    }

    fn select_node(&mut self) -> &MCNode<A, B> {
        let mut node_id = 0;

        while !self.nodes[node_id].state.is_terminal() {
            if let Some(action) = self.find_free_action(node_id) {
                let expanded_id = self.expand(node_id, action);
                return &self.nodes[expanded_id];
            } else {
                node_id = self.best_child(node_id, 1f32).1;
            }
        }
        return &self.nodes[node_id];        
    }

    fn best_child(&self, node_id: usize, exploration_value: f32) -> (&B, usize) {
        let node = &self.nodes[node_id];

        let (best_child, _) = node.actions.iter()
        .filter(|&(_, a)| { // Remove actions without a child
            match a.child_id {
                Some(_) => true,
                None => false
            }
        })
        .fold((None, std::f32::MIN), |acc, (a, ma)| {
            let qsa = ma.action_value / ma.action_count as f32;
            let usa = exploration_value * ma.probability * (node.visit_count as f32).sqrt() / (1f32 + ma.action_value);
            let value = qsa + usa;

            if value > acc.1 {
                (Some((a, ma)), value)
            } else {
                acc
            }            
        });

        let (best_action, best_child) = best_child.unwrap();

        (best_action, best_child.child_id.unwrap())
    }

    fn find_free_action(&mut self, node_id: usize) -> Option<B> {
        let node = &self.nodes[node_id];

        // Find an action that has no child yet (action has not yet been done)
        node.actions.iter().find(|(_, s)| match s.child_id {
            Some(_) => false,
            None => true
        }).map(|s| s.0.clone())
    }


    fn expand(&mut self, node_id: usize, action: B) -> usize {                    
        
        // Set child id of node.
        self.nodes[node_id].actions.get_mut(&action).unwrap().child_id = Some(self.nodes.len());

         // Take this action and get the next state
        let next_state = self.nodes[node_id].state.take_action(action);
       
        
        let next_node = MCNode::evaluate(next_state, Some(node_id), &self.model, {
            // Build the game history needed for evaluation
            let mut history = Vec::new();
            let mut parent = node_id; //Start at node
            while let Some(e) = self.nodes[parent].parent {
                parent = e; // Go to my parent
                history.push(parent);
            }   
            history.into_iter().map(|id| {
                &self.nodes[id].state
            }).collect()
        });
        

        self.nodes.push(next_node);        

        self.nodes.len() - 1
    }
}

struct MCAction{child_id: Option<usize>, action_count: u32, action_value: f32, probability: f32}
struct MCNode<A, B> where A: GameState<B>, B: GameAction {
    state: A,
    parent: Option<usize>,
    win_factor: f32,
    visit_count: u32,
    actions: HashMap<B, MCAction>
}

impl<A, B> MCNode<A, B> where A: GameState<B>, B: GameAction {
    fn reward(&self) -> f32 {
        if self.state.is_terminal() {
            self.state.terminal_reward()
        } else {
            self.win_factor
        }
    }
}

impl<A, B> MCNode<A, B> where A: GameState<B>, B: GameAction {    

    fn evaluate(state: A, parent: Option<usize>, model: &CatZeroModel, history: Vec<&A>) -> Self {
        
        let (win_factor, action_probs) = model.evaluate(state.into_tensor(history)).unwrap();
        
        let possible_actions = state.possible_actions();
        let action_probs = A::into_actions(action_probs).into_iter() 
            .filter(|(_, action)| possible_actions.contains(action) ) // Keep only possible probabilities
            .map(|(prob, action)| {
                (action, MCAction {
                    child_id: None,
                    action_count: 0,
                    action_value: 0f32,
                    probability: prob
                })
            }).collect();
        
        MCNode {
            state, parent, win_factor,
            visit_count: 0,
            actions: action_probs
        }
    }
}
