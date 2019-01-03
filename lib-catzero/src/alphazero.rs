use crate::model::CatZeroModel;
use crate::game::*;

pub struct AlphaZero<'a, G, A, B, C, D> where G: Game<A, B, C, D>, A: GameAction, B: GameState<A>, C: Agent<A, B>, D: Agent<A, B> {
    best_model: CatZeroModel<'a>,
    game: G,
    player_1: C,
    player_2: D,
    history: Vec<B>,
    latest_state: B,
    phantom: std::marker::PhantomData<A>
}

impl<'a, G, A, B, C, D> AlphaZero<'a, G, A, B, C, D> where G: Game<A, B, C, D>, A: GameAction, B: GameState<A>, C: Agent<A, B>, D: Agent<A, B> {

}