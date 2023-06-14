#![allow(dead_code)]

use std::path::PathBuf;

use cxx::{let_cxx_string, UniquePtr};
use ffi::{CTranslateOptions, MyDataClass};

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

pub struct TranslationOptions {
    /// Beam size to use for beam search (set 1 to run greedy search).
    pub beam_size: usize,
    /// Beam search patience factor, as described in https:///arxiv.org/abs/2204.05424.
    /// The decoding will continue until beam_size*patience hypotheses are finished.
    pub patience: f32,
    /// Exponential penalty applied to the length during beam search.
    /// The scores are normalized with:
    ///   hypothesis_score /= (hypothesis_length ** length_penalty)
    pub length_penalty: f32,
    /// Coverage penalty weight applied during beam search.
    pub coverage_penalty: f32,
    /// Penalty applied to the score of previously generated tokens, as described in
    /// https:///arxiv.org/abs/1909.05858 (set > 1 to penalize).
    pub repetition_penalty: f32,
    /// Prevent repetitions of ngrams with this size (set 0 to disable).
    pub no_repeat_ngram_size: usize,
    /// Disable the generation of the unknown token.
    pub disable_unk: bool,
    /// Biases decoding towards a given prefix, see https:///arxiv.org/abs/1912.03393 --section 4.2
    /// Only activates biased-decoding when beta is in range (0, 1) and SearchStrategy is set to BeamSearch.
    /// The closer beta is to 1, the stronger the bias is towards the given prefix.
    ///
    /// If beta <= 0 and a non-empty prefix is given, then the prefix will be used as a
    /// hard-prefix rather than a soft, biased-prefix.
    pub prefix_bias_beta: f32,

    /// Include the end token in the result.
    pub return_end_token: bool,

    /// Truncate the inputs after this many tokens (set 0 to disable truncation).
    pub max_input_length: usize,

    /// Decoding length constraints.
    pub max_decoding_length: usize,
    pub min_decoding_length: usize,

    /// Randomly sample from the top K candidates (set 0 to sample from the full output distribution).
    pub sampling_topk: usize,
    /// High temperature increase randomness.
    pub sampling_temperature: f32,

    /// Allow using the vocabulary map included in the model directory, if it exists.
    pub use_vmap: bool,

    /// Number of hypotheses to store in the TranslationResult class.
    pub num_hypotheses: usize,

    /// Store scores in the TranslationResult class.
    pub return_scores: bool,
    /// Store attention vectors in the TranslationResult class.
    pub return_attention: bool,

    /// Return alternatives at the first unconstrained decoding position. This is typically
    /// used with a target prefix to provide alternatives at a specifc location in the
    /// translation.
    pub return_alternatives: bool,
    /// Minimum probability to expand an alternative.
    pub min_alternative_expansion_prob: f32,

    /// Replace unknown target tokens by the original source token with the highest attention.
    pub replace_unknowns: bool,
}

impl Default for TranslationOptions {
    fn default() -> Self {
        Self {
            beam_size: 2,
            patience: 1.0,
            length_penalty: 1.0,
            coverage_penalty: 0.0,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            disable_unk: false,
            prefix_bias_beta: 0.0,
            return_end_token: false,
            max_input_length: 1024,
            max_decoding_length: 256,
            min_decoding_length: 1,
            sampling_topk: 1,
            sampling_temperature: 1.0,
            use_vmap: false,
            num_hypotheses: 1,
            return_scores: false,
            return_attention: false,
            return_alternatives: false,
            min_alternative_expansion_prob: 0.0,
            replace_unknowns: false,
        }
    }
}

#[cxx::bridge()]
mod ffi {

    unsafe extern "C++" {
        include!("rustyctranslate2/include/translator.h");
        type MyTranslator;
        type MyDataClass;
        type CTranslateOptions;
        fn new_translator(
            model: &CxxString,
            use_gpu: bool,
            compressed: bool,
        ) -> Result<UniquePtr<MyTranslator>>;
        fn translate_batch(
            self: Pin<&mut MyTranslator>,
            data: &MyDataClass,
            options: &CTranslateOptions,
            max_batch_size: usize,
            batch_type_example: bool,
        ) -> Result<UniquePtr<MyDataClass>>;
        fn translate_batch_target(
            self: Pin<&mut MyTranslator>,
            data: &MyDataClass,
            target: Vec<String>,
            options: &CTranslateOptions,
            max_batch_size: usize,
            batch_type_example: bool,
        ) -> Result<UniquePtr<MyDataClass>>;
        fn new_data() -> UniquePtr<MyDataClass>;
        fn getLength(self: &MyDataClass) -> usize;
        fn pushData(self: Pin<&mut MyDataClass>, item: Vec<String>);
        fn getData(self: &MyDataClass, data: usize) -> Result<Vec<String>>;
        #[allow(clippy::too_many_arguments)]
        fn get_options(
            beam_size: usize,
            patience: f32,
            length_penalty: f32,
            coverage_penalty: f32,
            repetition_penalty: f32,
            no_repeat_ngram_size: usize,
            disable_unk: bool,
            prefix_bias_beta: f32,
            return_end_token: bool,
            use_vmap: bool,
            num_hypotheses: usize,
            return_scores: bool,
            return_attention: bool,
            return_alternatives: bool,
            min_alternative_expansion_prob: f32,
            replace_unknowns: bool,
            max_input_length: usize,
            max_decoding_length: usize,
            min_decoding_length: usize,
            sampling_topk: usize,
            sampling_temperature: f32,
        ) -> UniquePtr<CTranslateOptions>;
    }
}

unsafe impl Sync for ffi::MyTranslator {}
unsafe impl Sync for ffi::MyDataClass {}
unsafe impl Sync for ffi::CTranslateOptions {}

pub struct CTranslator {
    model: UniquePtr<MyTranslator>,
}

impl CTranslator {
    pub fn new(path: PathBuf, use_gpu: bool, compressed: bool) -> Result<Self, String> {
        let path = path.to_str().map(|v| v.to_string()).unwrap();
        let_cxx_string!(model = path);
        let model = ffi::new_translator(&model, use_gpu, compressed).map_err(|e| e.to_string())?;
        Ok(Self { model })
    }

    pub fn translate_batch(
        &mut self,
        input: Vec<Vec<String>>,
        max_batch_size: Option<usize>,
        options: Option<TranslationOptions>,
        batch_type: BatchType,
    ) -> Result<Vec<Vec<String>>, String> {
        let data = Self::generate_input(input)?;
        let options = self.get_options(options);
        let v = self
            .model
            .as_mut()
            .ok_or_else(|| "mut model is none".to_string())?
            .translate_batch(
                &data,
                &options,
                max_batch_size.unwrap_or(0),
                batch_type.to_bool(),
            )
            .map_err(|e| e.to_string())?;
        Self::extract_output(v)
    }

    pub fn translate_batch_target(
        &mut self,
        input: Vec<Vec<String>>,
        max_batch_size: Option<usize>,
        batch_type: BatchType,
        options: Option<TranslationOptions>,
        target: Vec<String>,
    ) -> Result<Vec<Vec<String>>, String> {
        let data = Self::generate_input(input)?;
        let options = self.get_options(options);
        let v = self
            .model
            .as_mut()
            .ok_or_else(|| "mut model is none".to_string())?
            .translate_batch_target(
                &data,
                target,
                &options,
                max_batch_size.unwrap_or(0),
                batch_type.to_bool(),
            )
            .map_err(|e| e.to_string())?;
        Self::extract_output(v)
    }

    fn generate_input(input: Vec<Vec<String>>) -> Result<UniquePtr<MyDataClass>, String> {
        let mut data = ffi::new_data();
        for item in input {
            data.as_mut()
                .ok_or_else(|| "mut mydataclass is none".to_string())?
                .pushData(item);
        }
        Ok(data)
    }

    fn extract_output(v: UniquePtr<MyDataClass>) -> Result<Vec<Vec<String>>, String> {
        let mut res = vec![];
        let length = v.getLength();
        for index in 0..length {
            res.push(v.getData(index).map_err(|e| e.to_string())?);
        }
        Ok(res)
    }

    fn get_options(&self, options: Option<TranslationOptions>) -> UniquePtr<CTranslateOptions> {
        let o = options.unwrap_or_default();
        ffi::get_options(
            o.beam_size,
            o.patience,
            o.length_penalty,
            o.coverage_penalty,
            o.repetition_penalty,
            o.no_repeat_ngram_size,
            o.disable_unk,
            o.prefix_bias_beta,
            o.return_end_token,
            o.use_vmap,
            o.num_hypotheses,
            o.return_scores,
            o.return_attention,
            o.return_alternatives,
            o.min_alternative_expansion_prob,
            o.replace_unknowns,
            o.max_input_length,
            o.max_decoding_length,
            o.min_decoding_length,
            o.sampling_topk,
            o.sampling_temperature,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    #[test]
    fn it_works() {
        let model = CTranslator::new(
            PathBuf::from_str("jparacrawl/base-en-ja").unwrap(),
            false,
            true,
        );
        assert!(model.is_ok());
        let tokens = ["▁H", "ell", "o", "▁world", "!"]
            .into_iter()
            .map(|v| v.to_string())
            .collect();
        let v = model
            .unwrap()
            .translate_batch(vec![tokens], None, None, BatchType::Example);
        assert!(v.is_ok());
        println!("{:?}", v);
    }
}
