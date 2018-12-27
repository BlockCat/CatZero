use crate::mcts::*;

enum BoardStone {
    Cross, Circle
}

struct Board {
    stones: Option<BoardStone>
}

struct TTTMove {

}

impl Move for TTTMove {

}

impl Policy<TTTMove> for Board {

    fn policy(&self, moves: Vec<TTTMove>) -> Vec<f32> {
        vec!()
        // according to alpha zero...
        // a value 
    }
    
}