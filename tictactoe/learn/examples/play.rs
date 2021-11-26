use catzero::{CatZeroModel, TFModel};
use mcts::{
    transposition_table::ApproxTable,
    tree_policy::{AlphaGoPolicy, UCTPolicy},
    GameState, MCTSManager,
};
use tictactoe::TicTacToeAction;

include!("../src/lib.rs");

const EXPLORATION: f64 = 1.4;
const PLAYOUTS: usize = 550;
const MODEL: usize = 130;

fn main() {
    let tf = TFModel::load(&format!("data/models/graph/{}", MODEL)).expect("Could not load model");

    let mut state = TicTacToeState::default();

    while !state.is_terminal() {
        
        let eval = tf.evaluate(state.clone().into()).expect("Could not validate");
        println!("{:?}", eval.value[0]);

        println!("{}", state);
        let ac = find_npc_action(&state, &tf);
        state.make_move(&ac);
    }

    println!("{}", state);
}

fn find_npc_action(state: &TicTacToeState, model: &TFModel) -> TicTacToeAction {
    let manager = TicTacToeMCTS::default();
    let policy = UCTPolicy::new(EXPLORATION);
    
    let mut mcts_manager = MCTSManager::new(
        state.clone(),
        manager.clone(),
        evaluator::MyEvaluator {
            winner: state.current_player(),
            model: &model,
        },
        policy.clone(),
        ApproxTable::new(1024),
    );    
    mcts_manager.playout_n(PLAYOUTS);
    // mcts_manager.tree().debug_moves();


    
    mcts_manager.principal_variation(1)[0].clone()
}
