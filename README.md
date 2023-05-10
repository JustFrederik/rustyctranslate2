# rustyctranslate2

A simple project that requires the ctranslate2 library. 
https://github.com/OpenNMT/CTranslate2
https://github.com/OpenNMT/CTranslate2/tree/master/python/tools

How to use?
```
let model = CTranslator::new(PathBuf::from_str("...").unwrap(), false);
let tokens = ["▁H", "ell", "o", "▁world", "!"].into_iter().map(|v| v.to_string()).collect();
let v = model.unwrap().translate_batch(vec![tokens], None, BatchType::Example);
println!("{:?}", v);
```
