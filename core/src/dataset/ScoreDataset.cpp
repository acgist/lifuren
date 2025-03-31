#include "lifuren/Dataset.hpp"

#include <map>
#include <iostream>

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/String.hpp"

static std::vector<std::vector<lifuren::dataset::score::Finger>> load_from_xml(const std::string&);
static std::vector<std::vector<lifuren::dataset::score::Finger>> load_from_txt(const std::string&);

std::vector<std::vector<lifuren::dataset::score::Finger>> lifuren::dataset::score::load_finger(const std::string& path) {
    const auto suffix = lifuren::file::file_suffix(path);
    if(suffix == ".txt") {
        return load_from_txt(path);
    } else if(suffix == ".xml" || suffix == ".musicxml") {
        return load_from_xml(path);
    } else {
        SPDLOG_WARN("不支持的文件格式：{}", path);
        return {};
    }
}

lifuren::dataset::SeqDatasetLoader lifuren::dataset::score::loadMozartDatasetLoader(const size_t batch_size, const std::string& path) {
    auto dataset = lifuren::dataset::Dataset(
        path,
        { ".txt", ".xml", ".musicxml" },
        [] (
            const std::string         & file,
            std::vector<torch::Tensor>& labels,
            std::vector<torch::Tensor>& features,
            const torch::DeviceType   & device
        ) {
            auto vectors = lifuren::dataset::score::load_finger(file);
            for(const auto vector : vectors) {
                std::vector<float> label;
                std::vector<float> feature;
                label.resize(5);
                // 1 + 12 + 10 + 1 = 24
                feature.resize(24 * 5);
                for(int i = 0; i < vector.size(); ++i) {
                    const auto value = vector[i];
                    std::fill(label.begin(),   label.end(),   0.0F);
                    std::fill(feature.begin(), feature.end(), 0.0F);
                    label[value.finger - 1]    = 1.0F;
                    feature[0]                 = value.duration;
                    feature[value.step]        = 1.0F;
                    feature[value.octave + 13] = 1.0F;
                    feature[23]                = value.hand;
                    for(int j = 1; j < 5 && i + j < vector.size(); ++j) {
                        const auto next = vector[i + j];
                        feature[j * 24 + 0]                = next.duration;
                        feature[j * 24 + next.step]        = 1.0F;
                        feature[j * 24 + next.octave + 13] = 1.0F;
                        feature[j * 24 + 23]               = next.hand;
                    }
                    auto label_tensor   = torch::from_blob(label.data(),   {     5 }, torch::kFloat32);
                    auto feature_tensor = torch::from_blob(feature.data(), { 5, 24 }, torch::kFloat32);
                    labels.push_back(label_tensor.clone().to(device));
                    features.push_back(feature_tensor.clone().to(device));
                }
            }
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<LFT_SEQ_SAMPLER>(std::move(dataset), batch_size);
}

static std::vector<std::vector<lifuren::dataset::score::Finger>> load_from_xml(const std::string& file) {
    return {};
}

static std::vector<std::vector<lifuren::dataset::score::Finger>> load_from_txt(const std::string& file) {
    std::ifstream stream;
    stream.open(file);
    if(!stream.is_open()) {
        SPDLOG_WARN("文件打开失败：{}", file);
        return {};
    }
    std::string line;
    std::vector<lifuren::dataset::score::Finger> fingers_r;
    std::vector<lifuren::dataset::score::Finger> fingers_l;
    fingers_r.reserve(1024);
    fingers_l.reserve(1024);
    while(std::getline(stream, line)) {
        if(line.empty()) {
            continue;
        }
        auto vector = lifuren::string::split(line, std::vector<std::string>{ " ", "\t" });
        if(vector.size() != 8) {
            continue;
        }
        static std::map<std::string, int> step_map{
            {"C",  1 },
            {"C#", 2 },
            {"Db", 2 },
            {"D",  3 },
            {"D#", 4 },
            {"Eb", 4 },
            {"E",  5 },
            {"F",  6 },
            {"F#", 7 },
            {"Gb", 7 },
            {"G",  8 },
            {"G#", 9 },
            {"Ab", 9 },
            {"A",  10},
            {"A#", 11},
            {"Bb", 11},
            {"B",  12},
        };
        auto pitch = vector[3];
        int pitch_length = pitch.length();
        auto iter = step_map.find(pitch.substr(0, pitch_length - 1));
        if(iter == step_map.end()) {
            SPDLOG_WARN("不支持的符号：{}", pitch);
            return {};
        }
        auto finger = std::atoi(vector[7].c_str());
        if(finger < 0) {
            fingers_l.push_back({
                .duration = static_cast<float>(std::atof(vector[2].c_str()) - std::atof(vector[1].c_str())),
                .step   = iter->second,
                .octave = std::atoi(pitch.substr(pitch_length - 1).c_str()),
                .hand   = 0,
                .finger = std::abs(finger)
            });
        } else {
            fingers_r.push_back({
                .duration = static_cast<float>(std::atof(vector[2].c_str()) - std::atof(vector[1].c_str())),
                .step   = iter->second,
                .octave = std::atoi(pitch.substr(pitch_length - 1).c_str()),
                .hand   = 1,
                .finger = std::abs(finger)
            });
        }
    }
    return {fingers_r, fingers_l};
}
