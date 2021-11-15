use core::time;

use catzero::{CatZeroModel, Player, TFModel, Tensor};
use mcts::tree_policy::AlphaGoPolicy;
use mcts::{transposition_table::ApproxTable, Moves};
use mcts::{GameState, MCTSManager};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs};
use tictactoe::TicTacToeState;
use tmcts::TicTacToeMCTS;

use crate::tictactoe::TicTacToeAction;

mod evaluator;
mod tictactoe;
mod tmcts;

const EXPLORATION: f64 = 1.4;
const GAMES_TO_PLAY: usize = 100;
const PLAYOUTS: usize = 10;

const EPISODES: usize = 1;
const BATCH_SIZE: u32 = 10;
const EPOCHS: u32 = 10;

fn main() {
    load_graph_model("tictactoe/models/graph");

    // let mut results = Vec::new();
    // for episode in 0..EPISODES {
    //     println!("Starting episode: {}", episode);
    //     for _ in 0..GAMES_TO_PLAY {
    //         println!("Starting a game");
    //         results.push(play_a_game(&nn1));
    //         println!("Played a game");
    //     }

    //     let inputs: Vec<Tensor<u8>> = results
    //         .iter()
    //         .flat_map(|result| result.histories.iter())
    //         .map(|(state, _)| state.into())
    //         .collect();

    //     println!("Collected: {} states in {} games, during episode {}", inputs.len(), GAMES_TO_PLAY, episode);

    //     let probs: Vec<Tensor<f32>> = results
    //         .iter()
    //         .flat_map(|result| result.histories.iter())
    //         .map(|(_, tensor)| tensor.clone())
    //         .collect();

    //     let rewards: Vec<f32> = results
    //         .iter()
    //         .map(|result| &result.winner)
    //         .map(|winner| match winner {
    //             Some(player) if player == &search_player => 1.0,
    //             Some(_) => -1.0,
    //             None => 0.0,
    //         })
    //         .collect();

    //     nn1.save("player_agent.h5").expect("Could not save");

    //     match nn1.learn(inputs, probs, rewards, BATCH_SIZE, EPOCHS) {
    //         Ok(_) => {}
    //         Err(e) => {
    //             println!("Errored learning: {:?}", e)
    //         }
    //     }
    // }
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
fn play_a_game(model: &CatZeroModel) -> GameResult {
    let mut state = TicTacToeState::default();
    let manager = tmcts::TicTacToeMCTS::default();
    let policy = AlphaGoPolicy::new(EXPLORATION);
    let evaluator = evaluator::MyEvaluator {
        winner: Player::Player1,
        model: &model,
    };
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

        mcts_manager.playout_n(PLAYOUTS as u64);

        let info = mcts_manager.principal_variation_info(1)[0];

        let child = info.child().unwrap();
        let evaluation = child.moves();

        histories.push((state.clone(), moves_to_tensor(evaluation)));

        state.make_move(info.get_move());
    }

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
