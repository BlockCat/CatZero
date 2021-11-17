use catzero::{CatZeroModel, TFModel};
use mcts::{transposition_table::ApproxTable, tree_policy::AlphaGoPolicy, GameState, MCTSManager};
use tictactoe::TicTacToeAction;

include!("../src/lib.rs");

const EXPLORATION: f64 = 10.4;
const GAMES_TO_PLAY: usize = 100;
const PLAYOUTS: usize = 200;

const EPISODES: usize = 100;
const BATCH_SIZE: u32 = 20;
const EPOCHS: u32 = 100;

fn main() {
    let mut pyenv = catzero::PyEnv::new();
    let python = pyenv.python();
    let model = CatZeroModel::load(&python, "data/models/graph", 12, (1, 3, 3)).expect("Could not load model");
    let tf = model
        .to_tf_model(std::usize::MAX)
        .expect("Could not create tf model");

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
