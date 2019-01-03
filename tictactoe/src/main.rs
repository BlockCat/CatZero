extern crate catzero;
extern crate hashbrown;

mod tictactoe;

use tictactoe::*;
use catzero::game::AlphaAgent;

fn main() {

    let mut pyenv = catzero::PyEnv::new();
    let python = pyenv.python();

    let nn1 = catzero::CatZeroModel::new(&python, (5, 3, 3), (1, 3, 3), 0.1, 5).expect("Could not create neural model");
    let nn2 = catzero::CatZeroModel::new(&python, (5, 3, 3), (1, 3, 3), 0.1, 5).expect("Could not create neural model");

    let agent1 = AlphaAgent::new(&nn1);
    let agent2 = PlayerAgent;//AlphaAgent::new(&nn2);    
    let mut ttt = TicTacToe::new(agent1, agent2);

    ttt.do_print(true);
    let winner = ttt.start();

    println!("{:?}", winner);
}