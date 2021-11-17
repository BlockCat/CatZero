#![feature(test)]

extern crate test;
use catzero::TFModel;

use self::test::Bencher;

include!("../src/lib.rs");

#[bench]
fn python(b: &mut Bencher) {
    let mut pyenv = catzero::PyEnv::new();
    let python = pyenv.python();
    let nn1 = catzero::CatZeroModel::new(&python, (3, 3, 3), (1, 3, 3), 0.001, 1.0, 3, String::from("models/bench"))
        .expect("Could not create new model");
    
    b.iter(|| {
        let state = crate::TicTacToeState::default();
        nn1.evaluate(state.into()).unwrap();
    });
}

#[bench]
fn tensorflow(b: &mut Bencher) {
    
    let model = TFModel::load("E:/Workspace/rust/CatZero/tictactoe/models/graph").expect("Could not load graph");
    
    b.iter(|| {
        let state = crate::TicTacToeState::default();
        model.evaluate(state.into()).unwrap();
    });
}
