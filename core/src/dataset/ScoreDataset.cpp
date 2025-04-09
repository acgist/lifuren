#include "lifuren/Dataset.hpp"

#include <map>
#include <iostream>

#include "tinyxml2.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/String.hpp"
#include "lifuren/Torch.hpp"

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

static void parse_part   (lifuren::dataset::score::Score&, tinyxml2::XMLElement*);
static void parse_measure(lifuren::dataset::score::Score&, tinyxml2::XMLElement*, std::map<int, lifuren::dataset::score::Staff>&);
static void parse_note   (lifuren::dataset::score::Score&, tinyxml2::XMLElement*, lifuren::dataset::score::Measure&);

static int parse_fifths(tinyxml2::XMLElement*);

lifuren::dataset::score::Score lifuren::dataset::score::load_xml(const std::string& path) {
    tinyxml2::XMLDocument doc;
    lifuren::dataset::score::Score score;
    score.file_path = path;
    if(doc.LoadFile(path.c_str()) != tinyxml2::XMLError::XML_SUCCESS) {
        SPDLOG_WARN("打开文件失败：{}", path);
        return score;
    }
    SPDLOG_DEBUG("打开文件：{}", path);
    auto root = doc.RootElement();
    auto part = root->FirstChildElement("part");
    while(part) {
        ::parse_part(score, part);
        part = part->NextSiblingElement("part");
    }
    return score;
}

bool lifuren::dataset::score::save_xml(const std::string& path, const lifuren::dataset::score::Score& score) {
    tinyxml2::XMLDocument doc;
    auto decl = doc.NewDeclaration();
    doc.InsertFirstChild(decl);
    auto root = doc.NewElement("score-partwise");
    root->SetAttribute("version", "4.0");
    doc.InsertEndChild(root);
    root->SetText("1234");
    if(doc.SaveFile(path.c_str()) == tinyxml2::XMLError::XML_SUCCESS) {
        SPDLOG_DEBUG("保存文件：{}", path);
        return true;
    } else {
        SPDLOG_WARN("保存文件失败：{}", path);
        return false;
    }
}

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
    std::vector<int> classify_size(6, 0);
    std::vector<std::vector<torch::Tensor>> labels_tensors;
    std::vector<std::vector<torch::Tensor>> features_tensors;
    auto dataset = lifuren::dataset::Dataset(
        path,
        { ".txt", ".xml", ".musicxml" },
        [batch_size, &classify_size, &labels_tensors, &features_tensors] (
            const std::string         & file,
            std::vector<torch::Tensor>& labels,
            std::vector<torch::Tensor>& features,
            const torch::DeviceType   & device
        ) {
            // 左手|右手 音高 八度
            // 2 + 12 + 10 = 24
            const static int feature_dims = 4;
            const auto vectors = lifuren::dataset::score::load_finger(file);
            for(const auto& vector : vectors) {
                if(vector.empty()) {
                    continue;
                }
                std::vector<float> label(6, 0.0F);
                // 前四 + 当前 + 后四
                std::vector<float> feature(feature_dims * 9, 0.0F);
                std::vector<torch::Tensor> label_tensors;
                std::vector<torch::Tensor> feature_tensors;
                for(size_t i = 0; i < vector.size(); ++i) {
                    const auto value = vector[i];
                    std::fill(label.begin(),   label.end(),   0.0F);
                    std::fill(feature.begin(), feature.end(), 0.0F);
                    label[value.finger] = 1.0F;
                    for(int j = 0; j < 9; ++j) {
                        if(i + j < 4 || i + j - 4 >= vector.size()) {
                            continue;
                        }
                        const auto next = vector[i + j - 4];
                        feature[j * feature_dims + 0] = 1.0F /  10 * (next.hand + 1);
                        feature[j * feature_dims + 1] = 1.0F /  10 * next.step;
                        feature[j * feature_dims + 2] = 1.0F /  10 * next.octave;
                        feature[j * feature_dims + 3] = 1.0F / 120 * (next.octave * 12 + next.step);
                    }
                    auto label_tensor   = torch::from_blob(label.data(),   { 6               }, torch::kFloat32);
                    auto feature_tensor = torch::from_blob(feature.data(), { 9, feature_dims }, torch::kFloat32);
                    // lifuren::logTensor("label", feature_tensor);
                    label_tensors.push_back(label_tensor.clone().to(device));
                    feature_tensors.push_back(feature_tensor.clone().to(device));
                }
                labels_tensors.push_back(std::move(label_tensors));
                features_tensors.push_back(std::move(feature_tensors));
                if(labels_tensors.size() >= batch_size) {
                    size_t sum = 0;
                    size_t max = 0;
                    size_t min = std::numeric_limits<size_t>::max();
                    for(const auto& value : features_tensors) {
                        sum += value.size();
                        max = std::max(max, value.size());
                        min = std::min(min, value.size());
                    }
                    size_t length = sum / labels_tensors.size();
                    // size_t length = (max + min) / 2;
                    for (size_t i = 0; i < length; i++) {
                        for(const auto& value : labels_tensors) {
                            if(i < value.size()) {
                                labels.push_back(value[i]);
                            } else {
                                static auto default_none = torch::tensor({ 1, 0, 0, 0, 0, 0 }, torch::kFloat32).to(device);
                                labels.push_back(default_none);
                            }
                            classify_size[labels.back().argmax(0).item<int>()] += 1;
                        }
                        for(const auto& value : features_tensors) {
                            if(i < value.size()) {
                                features.push_back(value[i]);
                            } else {
                                static auto default_none = torch::zeros({ 9, feature_dims }, torch::kFloat32).to(device);
                                features.push_back(default_none);
                            }
                        }
                    }
                    labels_tensors.clear();
                    features_tensors.clear();
                }
            }
        }
    ).map(torch::data::transforms::Stack<>());
    for(const auto& size : classify_size) {
        SPDLOG_DEBUG("分类数量：{}", size);
    }
    return torch::data::make_data_loader<LFT_SEQ_SAMPLER>(std::move(dataset), batch_size);
}

static std::vector<std::vector<lifuren::dataset::score::Finger>> load_from_xml(const std::string& file) {
    auto score = lifuren::dataset::score::load_xml(file);
    if(score.staffMap.size() != 1) {
        SPDLOG_DEBUG("不支持的乐谱格式：{} - {}", file, score.staffMap.size());
        return {};
    }
    auto staff_map = score.staffMap.begin()->second;
    if(staff_map.size() != 2) {
        SPDLOG_DEBUG("不支持的乐谱格式：{} - {}", file, staff_map.size());
        return {};
    }
    std::vector<lifuren::dataset::score::Finger> fingers_r;
    std::vector<lifuren::dataset::score::Finger> fingers_l;
    fingers_r.reserve(1024);
    fingers_l.reserve(1024);
    int index = 0;
    for(const auto& [k, staff] : staff_map) {
        for(const auto& measure : staff.measureList) {
            for(const auto& note : measure.noteList) {
                lifuren::dataset::score::Finger finger;
                auto step_iter = step_map.find(std::string(1, note.step));
                if(step_iter == step_map.end()) {
                    SPDLOG_WARN("不支持的符号：{}", note.step);
                    return {};
                }
                if(note.finger < 1 || note.finger > 5) {
                    SPDLOG_DEBUG("指法标记错误：{} - {} - {}", note.step, note.octave, note.finger);
                    continue;
                }
                finger.step   = step_iter->second + note.alter;
                finger.octave = note.octave;
                finger.finger = note.finger;
                if(finger.step < 1) {
                    finger.step   += 12;
                    finger.octave -= 1;
                } else if(finger.step > 12) {
                    finger.step   -= 12;
                    finger.octave += 1;
                }
                if(index == 0) {
                    finger.hand = 1;
                    fingers_r.push_back(std::move(finger));
                } else {
                    finger.hand = 0;
                    fingers_l.push_back(std::move(finger));
                }
            }
        }
        ++index;
    }
    return {fingers_l, fingers_r};
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
        auto pitch = vector[3];
        int pitch_length = pitch.length();
        auto step_iter = step_map.find(pitch.substr(0, pitch_length - 1));
        if(step_iter == step_map.end()) {
            SPDLOG_WARN("不支持的符号：{}", pitch);
            stream.close();
            return {};
        }
        auto finger = std::atoi(vector[7].c_str());
        if(finger < 0) {
            fingers_l.push_back({
                .hand   = 0,
                .step   = step_iter->second,
                .octave = std::atoi(pitch.substr(pitch_length - 1).c_str()),
                .finger = std::abs(finger)
            });
        } else {
            fingers_r.push_back({
                .hand   = 1,
                .step   = step_iter->second,
                .octave = std::atoi(pitch.substr(pitch_length - 1).c_str()),
                .finger = finger
            });
        }
    }
    stream.close();
    return {fingers_l, fingers_r};
}

static void parse_part(lifuren::dataset::score::Score& score, tinyxml2::XMLElement* element) {
    std::map<int, lifuren::dataset::score::Staff> map;
    std::string id = element->Attribute("id");
    auto measure = element->FirstChildElement("measure");
    while(measure) {
        parse_measure(score, measure, map);
        measure = measure->NextSiblingElement("measure");
    }
    score.staffMap.emplace(id, std::move(map));
}

static void parse_measure(lifuren::dataset::score::Score& score, tinyxml2::XMLElement* element, std::map<int, lifuren::dataset::score::Staff>& map) {
    std::map<int, lifuren::dataset::score::Measure> measureMap;
    auto note = element->FirstChildElement("note");
    while(note) {
        int staff_value = 1;
        auto staff = note->FirstChildElement("staff");
        if(staff) {
            staff_value = std::atoi(staff->GetText());
        }
        auto staff_iter = map.find(staff_value);
        if(staff_iter == map.end()) {
            staff_iter = map.emplace(staff_value, lifuren::dataset::score::Staff{}).first;
            staff_iter->second.fifths = parse_fifths(element);
        }
        auto measure_iter = measureMap.find(staff_value);
        if(measure_iter == measureMap.end()) {
            measure_iter = measureMap.emplace(staff_value, lifuren::dataset::score::Measure{}).first;
        }
        parse_note(score, note, measure_iter->second);
        note = note->NextSiblingElement("note");
    }
    for(auto& [k, v] : measureMap) {
        if(v.noteList.empty()) {
            continue;
        }
        map[k].measureList.push_back(std::move(v));
    }
}

static void parse_note(lifuren::dataset::score::Score& score, tinyxml2::XMLElement* element, lifuren::dataset::score::Measure& measure) {
    auto pitch = element->FirstChildElement("pitch");
    if(pitch) {
        lifuren::dataset::score::Note note;
        auto step   = pitch->FirstChildElement("step");
        auto alter  = pitch->FirstChildElement("alter");
        auto octave = pitch->FirstChildElement("octave");
        if(step) {
            note.step = step->GetText()[0];
        }
        if(alter) {
            note.alter = std::atoi(alter->GetText());
        }
        if(octave) {
            note.octave = std::atoi(octave->GetText());
        }
        auto notations = element->FirstChildElement("notations");
        if(notations) {
            auto technical = notations->FirstChildElement("technical");
            if(technical) {
                auto fingering = technical->FirstChildElement("fingering");
                note.finger = std::atoi(fingering->GetText());
            }
        }
        measure.noteList.push_back(std::move(note));
    }
}

static int parse_fifths(tinyxml2::XMLElement* element) {
    auto attributes = element->FirstChildElement("attributes");
    if(attributes) {
        auto key = attributes->FirstChildElement("key");
        if(key) {
            auto fifths = key->FirstChildElement("fifths");
            if(fifths) {
                return std::atoi(fifths->GetText());
            }
        }
    }
    return 0;
}
