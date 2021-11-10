use cpython::{Python, PyObject, PyModule, PyResult};

macro_rules! py_import {
    ($module:ident, $python:ident, from $modu:ident.$path:tt import $($name:tt), *) => {
        $(            
            $module.add(*$python, $name, $modu.get(*$python, $path)?.cast_into::<PyModule>(*$python)?.get(*$python, $name)?)?;
        )*
    }
}

pub type Tensor<A> = Vec<Vec<Vec<A>>>;

pub struct CatZeroModel<'a> {
    python: &'a Python<'a>,
    module: PyModule,
    model: PyObject,    
    output_shape: (u32, u32, u32)
}

impl<'a> CatZeroModel<'a> {

    // TODO: Make sure to change output shape into a tuple because thatś nicer to work with
    /// input_shape: (channels, dim, dim)
    pub fn new(python: &'a Python, input_shape: (u32, u32, u32), output_shape: (u32, u32, u32), reg_constant: f32, residual_blocks: u32) -> PyResult<Self> {

        let module = CatZeroModel::create_module(python)?;
        let model = module.call(*python, "create_model", ((python.None(), input_shape.0, input_shape.1, input_shape.2), output_shape.0 * output_shape.1 * output_shape.2, reg_constant, residual_blocks), None)?;

        Ok(CatZeroModel{ python, module, model, output_shape })
    }

    fn create_module(python: &Python) -> PyResult<PyModule> {
        let main_str = include_str!("../model/main.py");


        let module = PyModule::new(*python, "CatZero")?;

        // module.add(*python, "bb", python.import("builtins")?)?;
        // module.add(*python, "np", python.import("numpy")?)?;

        // let keras = python.import("keras").expect("Could not import keras module");
        // py_import!(module, python, from keras."models" import "Model", "load_model");
        // py_import!(module, python, from keras."layers" import "Input", "BatchNormalization", "Activation", "Add", "Dense", "Convolution2D", "Flatten");
        // py_import!(module, python, from keras."optimizers" import "SGD");
        // py_import!(module, python, from keras."regularizers" import "l2");
        
        // module.add(*python, "keras", keras)?;

        python.run(main_str, Some(&module.dict(*python)), None)?;        

        Ok(module)
    }

    pub fn evaluate(&self, tensor: Tensor<u8>) -> PyResult<(f32, Tensor<f32>)> {
        
        let pyres = self.module.call(*self.python, "evaluate", (tensor, &self.model, self.output_shape), None)
                .expect("Could not call python module: 'evaluate'");
        
        let (value, tensor): (f32, Tensor<f32>) = pyres.extract(*self.python)
                .expect("Could not extract rust types from python");

        Ok((value, tensor))
    }

    pub fn learn(&self, inputs: Vec<Tensor<u8>>, probs: Vec<Tensor<f32>>, rewards: Vec<f32>, batch_size: u32, epochs: u32) -> PyResult<()> {
        
        self.module.call(*self.python, "learn", (&self.model, inputs, probs, rewards, batch_size, epochs), None)?;

        Ok(())
    }

    pub fn load(python: &'a Python, path: &str, output_shape: (u32, u32, u32)) -> PyResult<CatZeroModel<'a>> {
        let path = {
            let mut path_buf = std::path::PathBuf::new();
            path_buf.push(std::env::current_dir().unwrap());
            path_buf.push(path);
            String::from(path_buf.to_str().unwrap())
        };

        let module = CatZeroModel::create_module(python)?;
        let model = module.call(*python, "load_model", (path, ), None)?;

        Ok(CatZeroModel { python, module, model, output_shape })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let mut path_buf = std::path::PathBuf::new();
        path_buf.push(std::env::current_dir().unwrap());
        path_buf.push(path);
              
        self.module.call(*self.python, "save_model", (path_buf.to_str().unwrap() , &self.model), None)?;

        Ok(())        
    }
}