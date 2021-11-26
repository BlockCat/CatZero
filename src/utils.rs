use std::sync::Arc;

use crate::{AlphaGame, TFModel};
use mcts::GameState;
use rand::prelude::SliceRandom;

// play a game and a list of states
pub fn play_a_game<A>(
    mut state: A::State,
    exploration: f64,
    playouts: usize,
    model: Arc<TFModel>,
) -> GameResult<A>
where
    A: AlphaGame,
    mcts::ThreadData<A>: Default,
{
    let mut rng = rand::thread_rng();

    let mut histories = Vec::new();

    while !state.is_terminal() {
        let mut mcts_manager =
            A::create_manager(state.clone(), exploration, playouts, model.clone());

        mcts_manager.playout_n(playouts);

        let root_node = mcts_manager.tree().root_node();
        let moves = root_node.moves().collect::<Vec<_>>();

        histories.push((state.clone(), A::moves_to_tensorflow(moves.clone())));

        let weighted_action = moves
            .choose_weighted(&mut rng, |i| i.visits())
            .expect("Could not get a random action");

        state.make_move(weighted_action.get_move());
    }

    GameResult::new(state.get_winner(), histories)
}

pub struct GameResult<A>
where
    A: AlphaGame,
{
    histories: Vec<(A::State, tensorflow::Tensor<f32>)>,
    winner: Option<mcts::Player<A>>,
}

impl<A> GameResult<A>
where
    A: AlphaGame,
{
    pub fn new(
        winner: Option<mcts::Player<A>>,
        histories: Vec<(A::State, tensorflow::Tensor<f32>)>,
    ) -> GameResult<A> {
        Self { histories, winner }
    }

    pub fn winner(&self) -> &Option<mcts::Player<A>> {
        &self.winner
    }

    pub fn history_entries(&self) -> &Vec<(A::State, tensorflow::Tensor<f32>)> {
        &self.histories
    }
}
