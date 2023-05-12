#pragma once

#include <iostream>

#include "ctranslate2/translator.h"

#include "rust/cxx.h"

class MyDataClass {
  public: MyDataClass(std::vector < std::vector < std::string >> data = {}): m_data(data) {}

  size_t getLength() const {
    return m_data.size();
  }

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
  const std::vector < std::vector < std::string >> get_all() const {
    return m_data;
  }

  // Function to push a std::vector<std::string> to the data
  void pushData(rust::Vec < rust::String > newData) {
    std::vector < std::string > item;
    for (const auto & str: newData) {
      item.push_back(std::string(str));
    }
    m_data.push_back(item);
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
    const size_t max_batch_size = 0,
      const bool batch_type_example = true) {
    const std::vector < std::vector < std::string >> batch(data.get_all());
    const auto options = ctranslate2::TranslationOptions();
    auto translation = m_translator.translate_batch(batch, options, max_batch_size,
      batch_type_example ? ctranslate2::BatchType::Examples :
      ctranslate2::BatchType::Tokens);
    return std::make_unique < MyDataClass > (extract(translation));
  }

  std::unique_ptr < MyDataClass > translate_batch_target(const MyDataClass & data, rust::Vec < rust::String > target,
    const size_t max_batch_size = 0,
      const bool batch_type_example = true) {
    std::vector < std::vector < std::string >> target_v;
    for (const auto & t: target) {
      target_v.push_back({
        std::string(t)
      });
    }
    const std::vector < std::vector < std::string >> batch(data.get_all());
    const auto options = ctranslate2::TranslationOptions();
    auto translation = m_translator.translate_batch(batch, target_v, options, max_batch_size,
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
