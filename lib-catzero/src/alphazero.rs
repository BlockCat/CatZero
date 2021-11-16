use crate::game::*;
use crate::model::Tensor;

pub struct AlphaZero<'a, A, B>
where
    A: GameAction + Sync,
    B: AlphaZeroState<A> + Sync,
{
    player_1: AlphaAgent<'a, A, B>,
    player_2: AlphaAgent<'a, A, B>,
}

// impl<'a, A, B> AlphaZero<'a, A, B>
// where
//     A: GameAction + Sync,
//     B: AlphaZeroState<A> + Sync,
// {
//     pub fn new(agent1: AlphaAgent<'a, A, B>, agent2: AlphaAgent<'a, A, B>) -> Self {
//         // Play match, player_1 is the play we are currently teaching, player_2 is the current best player
//         AlphaZero {
//             player_1: agent1,
//             player_2: agent2,
//         }
//     }

//     fn play_game(
//         &mut self,
//         player_first: Player,
//         player_second: Player,
//     ) -> (Option<Player>, Vec<(B, Tensor<f32>)>) {
//         let mut current_state = B::default();
//         let mut history = Vec::new();

//         while !current_state.is_terminal() {
//             let player_first = match player_first {
//                 Player::Player1 => &mut self.player_1,
//                 Player::Player2 => &mut self.player_2,
//             };
//             let (player_action, probs) = if let Some((player_action, probs)) =
//                 player_first.get_alpha_action(current_state.clone())
//             {
//                 (player_action, probs)
//             } else {
//                 break;
//             };

//             // Because the current state had new probabilities to learn, add it to history
//             history.push((current_state.clone(), probs));

//             current_state.make_move(&player_action);

//             if current_state.is_terminal() {
//                 break;
//             }

//             let player_second = match player_second {
//                 Player::Player1 => &mut self.player_1,
//                 Player::Player2 => &mut self.player_2,
//             };

//             let (player_action, probs) = if let Some((player_action, probs)) =
//                 player_second.get_alpha_action(current_state.clone())
//             {
//                 (player_action, probs)
//             } else {
//                 break;
//             };
//             history.push((current_state.clone(), probs));

//             current_state.make_move(&player_action);

//             println!("Finished cycle");
//         }

//         let winner = current_state.get_winner();
//         (winner, history)
//     }

//     pub fn start(&mut self, iterations: u8, matches: u8) {
//         // For so many iterations
//         for _ in 0..iterations {
//             for m in 0..matches {
//                 let (player_first, player_second, search_player) = if m % 2 == 0 {
//                     (Player::Player1, Player::Player2, Player::Player1)
//                 } else {
//                     (Player::Player2, Player::Player1, Player::Player1)
//                 };

//                 // retrieve winner and history
//                 let (winner, history) = self.play_game(player_first, player_second);

//                 let reward = search_player.reward(winner);
//                 let tensors: Vec<Tensor<u8>> = history
//                     .iter()
//                     .enumerate()
//                     .map(|(i, (state, _))| {
//                         state.into_tensor(history.iter().take(i).map(|(state, _)| state).collect())
//                     })
//                     .collect();

//                 let probs: Vec<Tensor<f32>> = history.into_iter().map(|(_, probs)| probs).collect();
//                 let rewards: Vec<f32> = vec![reward; probs.len()];

//                 // Learn game
//                 // data: tensor -> (move probabilities, final reward)
//                 self.player_1.learn(tensors, probs, rewards);
//             }

//             //let (winner, history) = self.play_game(&self.player_1, &self.player_2);

//             self.player_1.save("player_agent.h5");

//             // Play 3 matches to determine better player. In case of multiple draws choose newer guy
//         }
//     }
// }
