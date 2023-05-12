#pragma once

#include <iostream>

#include "ctranslate2/translator.h"

#include "rust/cxx.h"

class CTranslateOptions {
public: CTranslateOptions(ctranslate2::TranslationOptions d) : data(d) {}
  ctranslate2::TranslationOptions get() const {
    return data;
  }
private: ctranslate2::TranslationOptions data;
};

// convert vec<vec<string>> from rust to c++ and vice versa
class MyDataClass {
  public: MyDataClass(std::vector < std::vector < std::string >> data = {}): m_data(data) {}

  // gets amount of sub vectors
  size_t getLength() const {
    return m_data.size();
  }

  // gets sub vector
  rust::Vec < rust::String > getData(const size_t index) const {
    if (index >= m_data.size()) {
      throw std::out_of_range("Index out of range");
    }
    rust::Vec < rust::String > sentence;
    for (const auto & str: m_data[index]) {
      sentence.push_back(str);
    }
    return sentence;
  }

  // extract data from class in c++
  const std::vector < std::vector < std::string >> get_all() const {
    return m_data;
  }

  // Function to push a std::vector<std::string> from rust to the data
  void pushData(rust::Vec < rust::String > newData) {
    std::vector < std::string > item;
    for (const auto & str: newData) {
      item.push_back(std::string(str));
    }
    m_data.push_back(item);
  }

  // memory management
  ~MyDataClass() {
    for (auto& item : m_data) {
      item.clear();
    }
    m_data.clear();
  }

  private: std::vector < std::vector < std::string >> m_data;
};

class MyTranslator {
  public: MyTranslator(const std::string & model_path,
    const bool use_gpu, const bool fast): m_translator(std::string(model_path),
    use_gpu ? ctranslate2::Device::CUDA : ctranslate2::Device::CPU,
    fast ? (use_gpu ? ctranslate2::ComputeType::FLOAT16 : ctranslate2::ComputeType::INT8) : ctranslate2::ComputeType::DEFAULT, {
      0
    }, {}) {}

  std::unique_ptr < MyDataClass > translate_batch(const MyDataClass & data,
    const CTranslateOptions & options,
    const size_t max_batch_size = 0,
    const bool batch_type_example = true ) {
    const std::vector < std::vector < std::string >> batch(data.get_all());
    auto translation = m_translator.translate_batch(batch, options.get(), max_batch_size,
      batch_type_example ? ctranslate2::BatchType::Examples :
      ctranslate2::BatchType::Tokens);
    return std::make_unique < MyDataClass > (extract(translation));
  }

  std::unique_ptr < MyDataClass > translate_batch_target(const MyDataClass & data, rust::Vec < rust::String > target,
    const CTranslateOptions & options,
    const size_t max_batch_size = 0,
    const bool batch_type_example = true) {
    std::vector < std::vector < std::string >> target_v;
    for (const auto & t: target) {
      target_v.push_back({
        std::string(t)
      });
    }
    const std::vector < std::vector < std::string >> batch(data.get_all());
    auto translation = m_translator.translate_batch(batch, target_v, options.get(), max_batch_size,
      batch_type_example ? ctranslate2::BatchType::Examples :
      ctranslate2::BatchType::Tokens);
    return std::make_unique < MyDataClass > (extract(translation));
  }

  private: ctranslate2::Translator m_translator;

  std::vector < std::vector < std::string >> extract(const std::vector < ctranslate2::TranslationResult > translation) const {
    std::vector < std::vector < std::string >> result;
    for (const auto & t: translation) {
      std::vector < std::string > sentence;
      for (const auto & token: t.output()) {
        sentence.push_back(token);
      }
      result.push_back(sentence);
    }
    return result;
  }
};

std::unique_ptr < MyTranslator > new_translator(const std::string & model,
  const bool gpu, const bool fast) {
  return std::make_unique < MyTranslator > (model, gpu, fast);
}

std::unique_ptr < MyDataClass > new_data() {
  return std::make_unique < MyDataClass > ();
}

std::unique_ptr<CTranslateOptions> get_options(
    size_t beam_size = 2, float patience = 1, float length_penalty = 1,
    float coverage_penalty = 0, float repetition_penalty = 1,
    size_t no_repeat_ngram_size = 0, bool disable_unk = false,
    float prefix_bias_beta = 0, bool return_end_token = false,
    bool use_vmap = false, size_t num_hypotheses = 1,
    bool return_scores = false, bool return_attention = false,
    bool return_alternatives = false, float min_alternative_expansion_prob = 0,
    bool replace_unknowns = false, size_t max_input_length = 1024,
    size_t max_decoding_length = 256, size_t min_decoding_length = 1,
    size_t sampling_topk = 1, float sampling_temperature = 1) {
  auto v = ctranslate2::TranslationOptions();
  v.beam_size = beam_size;
  v.patience = patience;
  v.length_penalty = length_penalty;
  v.coverage_penalty = coverage_penalty;
  v.repetition_penalty = repetition_penalty;
  v.no_repeat_ngram_size = no_repeat_ngram_size;
  v.disable_unk = disable_unk;
  v.prefix_bias_beta = prefix_bias_beta;
  v.return_end_token = return_end_token;
  v.max_input_length = max_input_length;
  v.max_decoding_length = max_decoding_length;
  v.min_decoding_length = min_decoding_length;
  v.sampling_topk = sampling_topk;
  v.sampling_temperature = sampling_temperature;
  v.use_vmap = use_vmap;
  v.num_hypotheses = num_hypotheses;
  v.return_scores = return_scores;
  v.return_attention = return_attention;
  v.return_alternatives = return_alternatives;
  v.min_alternative_expansion_prob = min_alternative_expansion_prob;
  v.replace_unknowns = replace_unknowns;
  return std::make_unique<CTranslateOptions>(v);
}
