use core::time;

use catzero::{CatZeroModel, Player, Tensor};
use mcts::tree_policy::AlphaGoPolicy;
use mcts::{transposition_table::ApproxTable, Moves};
use mcts::{GameState, MCTSManager};
use tensorflow::{Graph, SavedModelBundle, SessionOptions, SessionRunArgs};
use tictactoe::TicTacToeState;
use tmcts::TicTacToeMCTS;

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

fn load_graph_model(path: &str) {
    let mut graph = Graph::new();
    let bundle = SavedModelBundle::load(&SessionOptions::new(), &["serve"], &mut graph, path)
        .expect("Could not load");

    let default_sig = bundle
        .meta_graph_def()
        .get_signature("serving_default")
        .unwrap();

    let input = default_sig
        .get_input("input_1")
        .expect("Could not get input");

    let policy_output = default_sig
        .get_output("policy_h")
        .expect("Could not get policy output");
    let value_output = default_sig
        .get_output("activation_10")
        .expect("Could not get activation?");
    // println!("Sigs: {:#?}", default_sig);
    println!("Name: {}", default_sig.method_name());
    println!("Inputs: {:?}: ", input);
    println!("Output policy: {:?}", policy_output);
    println!("Output value: {:?}", value_output);

    let op_input = graph
        .operation_by_name_required(&input.name().name)
        .expect("Could not get graph operation");

    let op_output_policy = graph
        .operation_by_name_required(&policy_output.name().name)
        .expect("Could not get policy output operation");

    let op_output_value = graph
        .operation_by_name_required(&value_output.name().name)
        .expect("Could not get value output operation");

    println!("op input: {:?}", op_input);
    println!("op output 1: {:?}", op_output_policy);
    println!("op output 2: {:?}", op_output_value);

    let input: tensorflow::Tensor<f32> = tensorflow::Tensor::new(&[1, 3, 3, 3]);

    {
        let mut evaluate_step = SessionRunArgs::new();

        evaluate_step.add_feed(&op_input, 0, &input);
        let token1 = evaluate_step.request_fetch(&op_output_value, 0);
        let token2 = evaluate_step.request_fetch(&op_output_policy, 1);

        bundle.session.run(&mut evaluate_step).expect("it to work");

        let a: f32 = evaluate_step.fetch(token1).expect("Could not get token1")[0];
        let b: tensorflow::Tensor<f32> = evaluate_step.fetch(token2).expect("Could not get token1");
        let b = b.into_iter().cloned().collect::<Vec<_>>();
        println!("A: {:?}", a);
        println!("B: {:?}", b);
    }
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
