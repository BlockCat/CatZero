use tensorflow::{Graph, SavedModelBundle, SessionOptions};
use tensorflow::{Result, SessionRunArgs, Tensor};

#[derive(Debug)]
pub struct TFModel {
    graph: Graph,
    bundle: SavedModelBundle,
    op_input: tensorflow::Operation,
    op_output_policy: tensorflow::Operation,
    op_output_value: tensorflow::Operation,
}

impl TFModel {
    pub fn load(path: &str) -> Result<Self> {
        let options = SessionOptions::new();
        let mut graph = Graph::new();
        let bundle = SavedModelBundle::load(&options, &["serve"], &mut graph, path)?;

        let default_sig = bundle
            .meta_graph_def()
            .get_signature("serving_default")
            .unwrap();

        let input = default_sig
            .get_input("input_1")
            .expect("Could not get input");

        let policy_output = default_sig
            .get_output("policy_h")
            .expect("Could not get policy output");
        let value_output = default_sig
            .get_output("activation_10")
            .expect("Could not get activation?");

        let op_input = graph
            .operation_by_name_required(&input.name().name)
            .expect("Could not get graph operation");

        let op_output_policy = graph
            .operation_by_name_required(&policy_output.name().name)
            .expect("Could not get policy output operation");

        let op_output_value = graph
            .operation_by_name_required(&value_output.name().name)
            .expect("Could not get value output operation");

        println!("sigs: {:#?}", bundle.meta_graph_def().signatures());

        Ok(Self {
            graph,
            bundle,
            op_input,
            op_output_policy,
            op_output_value,
        })
    }

    pub fn evaluate(&self, input: Tensor<f32>) -> Result<TFEvaluation> {
        let mut session = SessionRunArgs::new();

        session.add_feed(&self.op_input, 0, &input);
        let token1 = session.request_fetch(&self.op_output_value, 0);
        let token2 = session.request_fetch(&self.op_output_policy, 1);

        self.bundle.session.run(&mut session)?;

        let value = session.fetch(token1)?;
        let policy = session.fetch(token2)?;

        Ok(TFEvaluation { policy, value })
    }

    // pub fn learn(&mut self, steps: usize) -> Result<()> {
    //     let me = self.bundle.meta_graph_def();

    //     let mut session = SessionRunArgs::new();

    //     session.add_feed(&self.op_input, 0, &input);

    //     for _ in 0..steps {
    //         self.bundle.session.run(&mut session);
    //     }

    //     Ok(())
    // }
}

pub struct TFEvaluation {
    pub policy: Tensor<f32>,
    pub value: Tensor<f32>,
}
