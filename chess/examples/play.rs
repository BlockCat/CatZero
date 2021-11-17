use catzero::{CatZeroModel, TFModel};
use mcts::{transposition_table::ApproxTable, tree_policy::AlphaGoPolicy, GameState, MCTSManager};
use tictactoe::TicTacToeAction;

include!("../src/lib.rs");

const EXPLORATION: f64 = 0.1;
const PLAYOUTS: usize = 200;

fn main() {
    let tf = TFModel::load("data/models/graph/9").expect("Could not load model");    

    let mut state = TicTacToeState::default();

    while !state.is_terminal() {
        println!("{}", state);
        let ac = find_npc_action(&state, &tf);
        state.make_move(&ac);
    }

    println!("{}", state);
}

fn find_npc_action(state: &TicTacToeState, model: &TFModel) -> TicTacToeAction {
    let manager = TicTacToeMCTS::default();
    let policy = AlphaGoPolicy::new(EXPLORATION);

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
    mcts_manager.principal_variation(1)[0].clone()
}
