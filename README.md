# rustyctranslate2

A simple project that requires the ctranslate2 library. 
https://github.com/OpenNMT/CTranslate2
https://github.com/OpenNMT/CTranslate2/tree/master/python/tools

How to use?
```
let model = CTranslator::new("...", false);
let v = model.unwrap().translate_batch(vec![vec!["Hello world!".to_string()]],None,BatchType::Example);
println!("{:?}", v);
```
