#include "lifuren/Audio.hpp"

#include <fstream>

#include "lifuren/File.hpp"
#include "lifuren/AudioModel.hpp"

namespace lifuren::audio {

template<typename M>
using AudioModelClientImpl = ModelClientImpl<lifuren::config::ModelParams, std::string, std::string, M>;

template<typename M>
class AudioClient : public AudioModelClientImpl<M> {

public:
    std::tuple<bool, std::string> pred(const std::string& input) override;

};

};

template<>
std::tuple<bool, std::string> lifuren::audio::AudioClient<lifuren::audio::ShikuangModel>::pred(const std::string& input) {
    if(!this->model) {
        return { false, {} };
    }
    const auto output = lifuren::file::modify_filename(input, ".pcm", "gen");
    const auto [success, pcm_file] = lifuren::dataset::audio::toPcm(input);
    if(!success) {
        SPDLOG_WARN("PCM音频文件转换失败：{}", input);
        return { false, output };
    }
    std::ifstream input_stream;
    input_stream.open(pcm_file, std::ios_base::binary);
    if(!input_stream.is_open()) {
        SPDLOG_WARN("PCM音频文件打开失败：{}", pcm_file);
        input_stream.close();
        return { false, output };
    }
    std::ofstream output_stream;
    output_stream.open(output, std::ios_base::binary);
    if(!output_stream.is_open()) {
        SPDLOG_WARN("PCM音频文件打开失败：{}", output);
        input_stream.close();
        output_stream.close();
        return { false, output };
    }
    bool gen = false;
    std::vector<short> data;
    data.resize(LFR_AUDIO_PCM_LENGTH);
    std::vector<torch::Tensor> tensors;
    tensors.reserve(100);
    while(input_stream.read(reinterpret_cast<char*>(data.data()), LFR_AUDIO_PCM_LENGTH * sizeof(short))) {
        tensors.push_back(lifuren::dataset::audio::pcm_stft(data));
        if(tensors.size() == 100) {
            gen = true;
            auto result = this->model->pred(torch::cat(tensors));
                 result = result.contiguous();
            auto list   = result.split(1, 0);
            for(const auto& value : list) {
                auto pcm = lifuren::dataset::audio::pcm_istft(value);
                output_stream.write(reinterpret_cast<char*>(pcm.data()), pcm.size() * sizeof(short));
            }
            tensors.clear();
            tensors.reserve(100);
        }
    }
    input_stream.close();
    output_stream.close();
    if(!gen) {
        return { false, output };
    }
    return lifuren::dataset::audio::toFile(output);
};

std::unique_ptr<lifuren::audio::AudioModelClient> lifuren::audio::getAudioClient(const std::string& model) {
    if(model == "shikuang") {
        return std::make_unique<lifuren::audio::AudioClient<ShikuangModel>>();
    } else {
        return nullptr;
    }
}
