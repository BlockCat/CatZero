use cpython::{Python, PyObject, PyModule, PyResult};
use crate::game::{GameState, GameAction};

macro_rules! py_import {
    ($module:ident, $python:ident, from $modu:ident.$path:tt import $($name:tt), *) => {
        $(            
            $module.add(*$python, $name, $modu.get(*$python, $path)?.cast_into::<PyModule>(*$python)?.get(*$python, $name)?)?;
        )*
    }
}

pub struct CatZeroModel<'a> {
    python: &'a Python<'a>,
    module: PyModule,
    model: PyObject,
    input_shape: (u32, u32, u32), // Channel, dim, dim
    output_shape: (u32, u32, u32)
}

impl<'a> CatZeroModel<'a> {

    // TODO: Make sure to change output shape into a tuple because thatś nicer to work with
    pub fn new(python: &'a Python, input_shape: (u32, u32, u32), output_shape: (u32, u32, u32), reg_constant: f32, residual_blocks: u32) -> PyResult<Self> {

        let module = CatZeroModel::create_module(python)?;
        let model = module.call(*python, "create_model", ((python.None(), input_shape.0, input_shape.1, input_shape.2), output_shape.0 * output_shape.1 * output_shape.2, reg_constant, residual_blocks), None)?;

        Ok(CatZeroModel{ python, module, model, input_shape, output_shape })
    }

    pub fn evaluate(&self, tensor: Vec<Vec<Vec<u8>>>) -> PyResult<(f32, Vec<Vec<Vec<f32>>>)> {
        
        //self.module.call(*self.python, "evaluate", (path_buf.to_str().unwrap() , &self.model), None)?;

        //Ok(())   

        panic!()
    }

    pub fn load(python: &'a Python, path: &str, input_shape: (u32, u32, u32), output_shape: (u32, u32, u32)) -> PyResult<CatZeroModel<'a>> {
        let path = {
            let mut path_buf = std::path::PathBuf::new();
            path_buf.push(std::env::current_dir().unwrap());
            path_buf.push(path);
            String::from(path_buf.to_str().unwrap())
        };

        let module = CatZeroModel::create_module(python)?;
        let model = module.call(*python, "load_model", (path, ), None)?;

        Ok(CatZeroModel { python, module, model, input_shape, output_shape })
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let mut path_buf = std::path::PathBuf::new();
        path_buf.push(std::env::current_dir().unwrap());
        path_buf.push(path);
              
        self.module.call(*self.python, "save_model", (path_buf.to_str().unwrap() , &self.model), None)?;

        Ok(())        
    }


    fn create_module(python: &Python) -> PyResult<PyModule> {
        let main_str = include_str!("../model/main.py");
        let keras = python.import("keras")?;

        let module = PyModule::new(*python, "CatZero")?;
        
        module.add(*python, "bb", python.import("builtins")?)?;

        py_import!(module, python, from keras."models" import "Model", "load_model");
        py_import!(module, python, from keras."layers" import "Input", "BatchNormalization", "Activation", "Add", "Dense", "Convolution2D", "Flatten");
        py_import!(module, python, from keras."optimizers" import "SGD");
        py_import!(module, python, from keras."regularizers" import "l2");
        

        python.run(main_str, Some(&module.dict(*python)), None)?;

        Ok(module)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn module_loading() {
     //   let m = create_module(python: &Python)
    }
}