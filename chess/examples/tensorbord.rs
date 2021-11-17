use catzero::TrainingData;


fn main() {
    let td = TrainingData::load("data/11.games").expect("Could not get training data");
    let mut pyenv = catzero::PyEnv::new();
    let python = pyenv.python();
    let mut python_model = catzero::CatZeroModel::new(
        &python,
        (3, 3, 3),
        (1, 3, 3),
        0.001,
        1.0,
        3,
        String::from("data/models/graph"),
    )
    .expect("Could not create new model");

    python_model.learn(&td, 20, 100).expect("Could not learn");
}