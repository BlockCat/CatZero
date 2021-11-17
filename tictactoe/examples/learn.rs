use catzero::{TFModel, TrainingData};
use mcts::{transposition_table::ApproxTable, tree_policy::AlphaGoPolicy, GameState, MCTSManager};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

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
    let mut python_model = catzero::CatZeroModel::new(
        &python,
        (3, 3, 3),
        (1, 3, 3),
        0.001,
        1.0,
        3,
        String::from("data/models/graph"),
    )
    .expect("Could not create new model");

    let search_player = Player::Player1;

    for episode in 0..EPISODES {
        let model = python_model
            .to_tf_model(episode)
            .expect("Could not create tensor model");

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

        let data = TrainingData {
            inputs: inputs,
            output_policy: probs,
            output_value: rewards,
        };

        if let Err(e) = data.save(&format!("data/{}.games", episode)) {
            println!("Did not save game data: {}", e);
        }

        match python_model.learn(data, BATCH_SIZE, EPOCHS) {
            Ok(_) => {
                println!("Learned an episode: {}", episode)
            }
            Err(e) => {
                println!("Errored learning: {:?}", e);
                panic!("Failed learning")
            }
        }
    }
}

// play a game and a list of states
fn play_a_game(model: &TFModel) -> GameResult {
    let mut state = TicTacToeState::default();
    let manager = tmcts::TicTacToeMCTS::default();
    // let policy = AlphaGoPolicy::new(EXPLORATION);
    let policy = AlphaGoPolicy::new(EXPLORATION);

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
