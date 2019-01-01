extern crate cpython;
extern crate hashbrown;

mod pyenv;
mod model;
mod mcts;
mod game;

use pyenv::PyEnv;

fn main() {
    let mut env = PyEnv::new();
    let python = env.python();
    let sys = python.import("sys").unwrap();
    let version: String = sys.get(python, "version").unwrap().extract(python).unwrap();
    println!("Version: {}", version);    

    let nn = match model::CatZeroModel::new(&python, (5, 3, 3), (1, 3, 3), 0.1, 5) {
        Ok(e) => e,
        Err(e) => {
            e.print(python);
            return;
        }
    };

    //let result: String = nn.call(python, "create_model", cpython::NoArgs, None).unwrap().extract(python).unwrap();

    //println!("Result: {}", result);    
}