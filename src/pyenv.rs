
use cpython::{Python, PyDict, PyResult, GILGuard, NoArgs};

pub struct PyEnv<'a> {
    gil: GILGuard, 
    python: Option<Python<'a>>
}

impl<'a> PyEnv<'a> {
    pub fn new<'q>() -> PyEnv<'q> {
        let gil = Python::acquire_gil();

        PyEnv {
            gil: gil,
            python: None
        }
    }

    pub fn python(&'a mut self) -> Python<'a> {
        match self.python {
            Some(p) => return p,
            None => {
                self.python = Some(self.gil.python());
                return self.python.unwrap();
            }
        }
    }
}