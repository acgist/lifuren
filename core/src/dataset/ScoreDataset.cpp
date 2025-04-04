#include "lifuren/Dataset.hpp"

#include <map>
#include <iostream>

#include "tinyxml2.h"

#include "spdlog/spdlog.h"

#include "lifuren/File.hpp"
#include "lifuren/String.hpp"

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

bool lifuren::dataset::score::Score::empty() {
    return this->staffMap.empty();
}

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
    auto dataset = lifuren::dataset::Dataset(
        path,
        { ".txt", ".xml", ".musicxml" },
        [] (
            const std::string         & file,
            std::vector<torch::Tensor>& labels,
            std::vector<torch::Tensor>& features,
            const torch::DeviceType   & device
        ) {
            // 音高 八度 左手|右手
            // 12 + 10 + 1 = 23
            const static int feature_dims = 23;
            auto vectors = lifuren::dataset::score::load_finger(file);
            for(const auto& vector : vectors) {
                std::vector<float> label;
                std::vector<float> feature;
                label.resize(5);
                feature.resize(feature_dims * 5);
                for(size_t i = 0; i < vector.size(); ++i) {
                    const auto value = vector[i];
                    std::fill(label.begin(),   label.end(),   0.0F);
                    std::fill(feature.begin(), feature.end(), 0.0F);
                    label[value.finger - 1]    = 1.0F;
                    feature[value.step - 1]    = 1.0F;
                    feature[value.octave + 12] = 1.0F;
                    feature[22]                = value.hand;
                    for(size_t j = 1; j < 5 && i + j < vector.size(); ++j) {
                        const auto next = vector[i + j];
                        feature[j * feature_dims + next.step - 1]    = 1.0F;
                        feature[j * feature_dims + next.octave + 12] = 1.0F;
                        feature[j * feature_dims + 22]               = next.hand;
                    }
                    auto label_tensor   = torch::from_blob(label.data(),   { 5               }, torch::kFloat32);
                    auto feature_tensor = torch::from_blob(feature.data(), { 5, feature_dims }, torch::kFloat32);
                    labels.push_back(label_tensor.clone().to(device));
                    features.push_back(feature_tensor.clone().to(device));
                }
            }
        }
    ).map(torch::data::transforms::Stack<>());
    return torch::data::make_data_loader<LFT_SEQ_SAMPLER>(std::move(dataset), batch_size);
}

static std::vector<std::vector<lifuren::dataset::score::Finger>> load_from_xml(const std::string& file) {
    auto score = lifuren::dataset::score::load_xml(file);
    // score.staffMap.
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
        auto pitch = vector[3];
        int pitch_length = pitch.length();
        auto iter = step_map.find(pitch.substr(0, pitch_length - 1));
        if(iter == step_map.end()) {
            SPDLOG_WARN("不支持的符号：{}", pitch);
            stream.close();
            return {};
        }
        auto finger = std::atoi(vector[7].c_str());
        if(finger < 0) {
            fingers_l.push_back({
                .step   = iter->second,
                .octave = std::atoi(pitch.substr(pitch_length - 1).c_str()),
                .hand   = 0,
                .finger = std::abs(finger)
            });
        } else {
            fingers_r.push_back({
                .step   = iter->second,
                .octave = std::atoi(pitch.substr(pitch_length - 1).c_str()),
                .hand   = 1,
                .finger = finger
            });
        }
    }
    stream.close();
    return {fingers_r, fingers_l};
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
