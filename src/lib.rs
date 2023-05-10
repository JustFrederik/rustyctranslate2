#![allow(dead_code)]

use cxx::{let_cxx_string, UniquePtr};
use std::path::PathBuf;

use crate::ffi::MyTranslator;

#[derive(Default)]
pub enum BatchType {
    #[default]
    Example,
    Tokens,
}

impl BatchType {
    fn to_bool(&self) -> bool {
        matches!(self, BatchType::Example)
    }
}

#[cxx::bridge()]
mod ffi {
    unsafe extern "C++" {
        include!("rustyctranslate2/include/translator.h");
        type MyTranslator;
        type MyDataClass;
        fn new_translator(model: &CxxString, use_gpu: bool) -> Result<UniquePtr<MyTranslator>>;
        fn translate_batch(
            self: Pin<&mut MyTranslator>,
            data: &MyDataClass,
            max_batch_size: usize,
            batch_type_example: bool,
        ) -> Result<UniquePtr<MyDataClass>>;
        fn translate_batch_target(
            self: Pin<&mut MyTranslator>,
            data: &MyDataClass,
            target: Vec<String>,
            max_batch_size: usize,
            batch_type_example: bool,
        ) -> Result<UniquePtr<MyDataClass>>;
        fn new_data() -> UniquePtr<MyDataClass>;
        fn getLength(self: &MyDataClass) -> usize;
        fn pushData(self: Pin<&mut MyDataClass>, item: Vec<String>);
        fn getData(self: &MyDataClass, data: usize) -> Result<Vec<String>>;
    }
}

pub struct CTranslator {
    model: UniquePtr<MyTranslator>,
}

impl CTranslator {
    pub fn new(path: PathBuf, use_gpu: bool) -> Result<Self, String> {
        let path = path.to_str().map(|v| v.to_string()).unwrap();
        let_cxx_string!(model = path);
        let model = ffi::new_translator(&model, use_gpu).map_err(|e| e.to_string())?;
        Ok(Self { model })
    }

    pub fn translate_batch(
        &mut self,
        input: Vec<Vec<String>>,
        max_batch_size: Option<usize>,
        batch_type: BatchType,
    ) -> Result<Vec<Vec<String>>, String> {
        let mut data = ffi::new_data();
        for item in input {
            data.as_mut()
                .ok_or_else(|| "mut mydataclass is none".to_string())?
                .pushData(item);
        }
        let v = self
            .model
            .as_mut()
            .ok_or_else(|| "mut model is none".to_string())?
            .translate_batch(&data, max_batch_size.unwrap_or(0), batch_type.to_bool())
            .map_err(|e| e.to_string())?;
        let mut res = vec![];
        let length = v.getLength();
        for index in 0..length {
            res.push(v.getData(index).map_err(|e| e.to_string())?);
        }
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    #[test]
    fn it_works() {
        let model = CTranslator::new(PathBuf::from_str("jparacrawl/base-en-ja").unwrap(), false);
        assert!(model.is_ok());
        let tokens = ["▁H", "ell", "o", "▁world", "!"]
            .into_iter()
            .map(|v| v.to_string())
            .collect();
        let v = model
            .unwrap()
            .translate_batch(vec![tokens], None, BatchType::Example);
        assert!(v.is_ok());
        println!("{:?}", v);
    }
}
