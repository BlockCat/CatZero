extern crate catzero;
extern crate hashbrown;

mod tictactoe;

use tictactoe::*;
use catzero::game::AlphaAgent;
use catzero::AlphaZero;

fn main() {

    let mut pyenv = catzero::PyEnv::new();
    let python = pyenv.python();
    
    let nn1 = catzero::CatZeroModel::load(&python, "player_agent.h5", (1, 3, 3)).unwrap();
/*
    let nn1 = catzero::CatZeroModel::new(&python, (5, 3, 3), (1, 3, 3), 0.1, 5).expect("Could not create neural model");
    //let nn2 = catzero::CatZeroModel::new(&python, (5, 3, 3), (1, 3, 3), 0.1, 5).expect("Could not create neural model");

    let agent1 = AlphaAgent::new(&nn1, 1f32);
    let agent2 = AlphaAgent::new(&nn1, 1f32);    

    let mut alphazero = AlphaZero::<TicTacToeAction, TicTacToeState>::new(agent1, agent2);
    for _ in 0..2 {
        alphazero.start(1, 10);
        let mut ttt = TicTacToe::new(AlphaAgent::new(&nn1, 1f32), AlphaAgent::new(&nn1, 1f32));
        
        ttt.do_print(true);
        ttt.start();
        nn1.save("player_agent.h5").unwrap();
    }*/

    let mut ttt = TicTacToe::new(AlphaAgent::new(&nn1, 1f32), AlphaAgent::new(&nn1, 1f32));
        
    ttt.do_print(true);
    let winner = ttt.start();

    println!("{:?}", winner);
}