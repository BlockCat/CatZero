use catzero::{CatZeroModel, TFModel, TrainingData};
use mcts::{
    transposition_table::ApproxTable,
    tree_policy::{AlphaGoPolicy, UCTPolicy},
    GameState, MCTSManager,
};
use tictactoe::TicTacToeAction;

include!("../src/lib.rs");

const EXPLORATION: f64 = 3.4;
const PLAYOUTS: usize = 550;

fn main() {
    let data =
        TrainingData::load(&format!("data/{}.games", 29)).expect("Could not find trainingdata");

    data.print(0..data.len());

    // let mut state = TicTacToeState::default();
    // state.make_move(&TicTacToeAction{x: 1, y: 0});
    // state.make_move(&TicTacToeAction{x: 2, y: 0});
    // state.make_move(&TicTacToeAction{x: 1, y: 1});
    // state.make_move(&TicTacToeAction{x: 1, y: 2});
    // state.make_move(&TicTacToeAction{x: 0, y: 2});
    // state.make_move(&TicTacToeAction{x: 2, y: 2});

    // println!("{}", state);

    test_playout();

}
//| 
//| O
//|XOX

fn test_playout() {

    let tf = TFModel::load("data/models/graph/30").expect("Could not load model");
    let mut state = TicTacToeState::default();
    state.make_move(&TicTacToeAction{x: 0, y: 0});
    // state.make_move(&TicTacToeAction{x: 2, y: 0});
    // state.make_move(&TicTacToeAction{x: 1, y: 1});
    // state.make_move(&TicTacToeAction{x: 1, y: 2});
    // state.make_move(&TicTacToeAction{x: 0, y: 2});
    // state.make_move(&TicTacToeAction{x: 2, y: 2});

    println!("{}", state);
    

    let manager = TicTacToeMCTS::default();
    let policy = UCTPolicy::new(EXPLORATION);

    let mut mcts_manager = MCTSManager::new(
        state.clone(),
        manager.clone(),
        evaluator::MyEvaluator {
            winner: state.current_player(),
            model: &tf,
        },
        policy.clone(),
        ApproxTable::new(1024),
    );

    mcts_manager.playout_n(PLAYOUTS);

    println!("ROOT State");
    mcts_manager.tree().debug_moves();

}
