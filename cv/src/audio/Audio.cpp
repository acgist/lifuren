#include "lifuren/audio/Audio.hpp"

#include <fstream>

#include "lifuren/File.hpp"
#include "lifuren/audio/AudioModel.hpp"

template<typename M>
std::tuple<bool, std::string> lifuren::audio::AudioClient<M>::pred(const AudioParams& input) {
    const auto output = lifuren::file::modify_filename(input.audio, ".pcm", "gen");
    const auto [success, pcm_file] = lifuren::audio::toPcm(input.audio);
    if(!success) {
        SPDLOG_WARN("PCM音频文件转换失败：{}", input.audio);
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
    std::vector<short> data;
    data.resize(DATASET_PCM_LENGTH);
    std::vector<torch::Tensor> tensors;
    tensors.reserve(100);
    while(input_stream.read(reinterpret_cast<char*>(data.data()), DATASET_PCM_LENGTH * sizeof(short))) {
        auto tensor = lifuren::audio::pcm_stft(data);
        tensors.push_back(tensor);
        if(tensors.size() == 100) {
            auto result = this->model->pred(torch::cat(tensors));
                 result = result.contiguous();
            auto list   = result.split(1, 0);
            for(const auto& value : list) {
                auto pcm = lifuren::audio::pcm_istft(value);
                output_stream.write(reinterpret_cast<char*>(pcm.data()), pcm.size() * sizeof(short));
            }
            tensors.clear();
            tensors.reserve(100);
        }
    }
    input_stream.close();
    output_stream.close();
    return lifuren::audio::toFile(output);
};

std::unique_ptr<lifuren::audio::AudioModelClient> lifuren::audio::getAudioClient(
    const std::string& model
) {
    if(model == lifuren::config::CONFIG_AUDIO_SHIKUANG) {
        return std::make_unique<lifuren::audio::AudioClient<ShikuangModel>>();
    } else {
        return nullptr;
    }
}
