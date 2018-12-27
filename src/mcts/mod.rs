/// Implement monte carlo tree search
/// 
/// 

/// Implement this trait to calculate policies
/// A policy gives a probability to every move.
///
/// Input: vector of moves
/// Output: vector of probabilities
pub trait Policy<T> where T: Move {
    fn policy(&self, moves: Vec<T>) -> Vec<f32>;
}

pub trait Move {

}