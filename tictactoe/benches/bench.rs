#![feature(test)]

extern crate test;
use self::test::Bencher;

include!("../src/main.rs");

#[bench]
fn benchmark(b: &mut Bencher) {
    let mut pyenv = catzero::PyEnv::new();
    let python = pyenv.python();
    let nn1 = catzero::CatZeroModel::new(&python, (3, 3, 3), (1, 3, 3), 1.0, 3)
        .expect("Could not create new model");
    
    b.iter(|| {
        let state = crate::TicTacToeState::default();
        nn1.evaluate(state.into()).unwrap();
    });
}
