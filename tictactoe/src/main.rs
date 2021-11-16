use core::time;

use catzero::{CatZeroModel, Player, TFModel, Tensor};
use mcts::tree_policy::{AlphaGoPolicy, UCTPolicy};
use mcts::{transposition_table::ApproxTable, Moves};
use mcts::{GameState, MCTSManager};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs};
use tictactoe::TicTacToeState;
use tmcts::TicTacToeMCTS;

use crate::tictactoe::TicTacToeAction;

mod evaluator;
mod tictactoe;
mod tmcts;

const EXPLORATION: f64 = 10.4;
const GAMES_TO_PLAY: usize = 60;
const PLAYOUTS: usize = 200;

const EPISODES: usize = 10;
const BATCH_SIZE: u32 = 10;
const EPOCHS: u32 = 100;

fn main() {
    let mut pyenv = catzero::PyEnv::new();
    let python = pyenv.python();
    let mut python_model = catzero::CatZeroModel::new(
        &python,
        (3, 3, 3),
        (1, 3, 3),
        0.001,
        1.0,
        3,
        String::from("models/graph"),
    )
    .expect("Could not create new model");

    
    let search_player = Player::Player1;

    for episode in 0..EPISODES {
        let model = python_model.to_tf_model(episode).expect("Could not create tensor model");
    
        // let mut results = Vec::new();
        println!("Starting episode: {}", episode);

        let results = (0..GAMES_TO_PLAY)
            .into_par_iter()
            .map(|i| {
                println!("Starting a game: {}", i);
                let res = play_a_game(&model);
                println!("Played a game: {}", i);
                res
            })
            .collect::<Vec<_>>();

        let inputs: Vec<Tensor<u8>> = results
            .iter()
            .flat_map(|result| result.histories.iter())
            .map(|(state, _)| state.into())
            .collect();

        println!(
            "Collected: {} states in {} games, during episode {}",
            inputs.len(),
            GAMES_TO_PLAY,
            episode
        );

        let probs: Vec<Tensor<f32>> = results
            .iter()
            .flat_map(|result| result.histories.iter())
            .map(|(_, tensor)| tensor.clone())
            .collect();

        let rewards: Vec<f32> = results
            .iter()
            .flat_map(|result| {
                let reward = match &result.winner {
                    Some(player) if player == &search_player => 1.0,
                    Some(_) => -1.0,
                    None => 0.0,
                };
                result.histories.iter().map(move |_| reward)
            })
            .collect();

        assert!(inputs.len() == probs.len());
        assert!(inputs.len() == rewards.len());

        match python_model.learn(inputs, probs, rewards, BATCH_SIZE, EPOCHS) {
            Ok(_) => { println!("Learned an episode: {}", episode)}
            Err(e) => {
                println!("Errored learning: {:?}", e);
                panic!("Failed learning")
            }
        }
    }
}

fn load_graph_model(path: &str) -> TFModel {
    let model = catzero::TFModel::load(path).expect("Could not load");

    // println!("Model: {:?}", model);

    // let input_tensor: tensorflow::Tensor<f32> = tensorflow::Tensor::new(&[1, 3, 3, 3]);
    let mut root_state = TicTacToeState::default();
    root_state.make_move(&TicTacToeAction { x: 0, y: 0 });
    let input_tensor = root_state.into();

    let ev = model.evaluate(input_tensor).expect("Could not validate");

    let a: f32 = ev.value[0];
    let b: Vec<f32> = ev.policy.iter().cloned().collect();
    println!("A: {:?}", a);
    println!("B: {:?}", b);

    model
}

// play a game and a list of states
fn play_a_game(model: &TFModel) -> GameResult {
    let mut state = TicTacToeState::default();
    let manager = tmcts::TicTacToeMCTS::default();
    // let policy = AlphaGoPolicy::new(EXPLORATION);
    let policy = AlphaGoPolicy::new(EXPLORATION);

    // let mut mcts_manager = MCTSManager::new(
    //     state.clone(),
    //     manager,
    //     evaluator,
    //     policy,
    //     ApproxTable::new(1024),
    // );

    let mut histories = Vec::new();

    while !state.is_terminal() {
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

        let info = mcts_manager.principal_variation_info(1)[0];

        let child = info.child().unwrap();
        let evaluation = child.moves();

        histories.push((state.clone(), moves_to_tensor(evaluation)));

        state.make_move(info.get_move());
    }

    println!("final: {}", state);

    GameResult::new(state.get_winner(), histories)
}

fn moves_to_tensor(moves: Moves<TicTacToeMCTS>) -> Tensor<f32> {
    let mut board = vec![vec![0.0; 3]; 3];

    moves.for_each(|info| {
        let action = info.get_move();
        board[action.x][action.y] = *info.move_evaluation() as f32;
    });

    vec![board]
}

struct GameResult {
    histories: Vec<(TicTacToeState, Tensor<f32>)>,
    winner: Option<Player>,
}

impl GameResult {
    pub fn new(
        winner: Option<Player>,
        histories: Vec<(TicTacToeState, Tensor<f32>)>,
    ) -> GameResult {
        Self { histories, winner }
    }
}
