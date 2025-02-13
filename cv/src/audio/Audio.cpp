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
    while(input_stream.read(reinterpret_cast<char*>(data.data()), DATASET_PCM_LENGTH * sizeof(short))) {
        auto tuple  = lifuren::audio::pcm_mag_pha_stft(data);
        auto result = this->model->pred(torch::cat({
            std::get<0>(tuple),
            std::get<1>(tuple)
        }).unsqueeze(0));
        auto list = result.squeeze().split(1, 0);
        auto pcm  = lifuren::audio::pcm_mag_pha_istft(
            list.at(0),
            list.at(1)
        );
        output_stream.write(reinterpret_cast<char*>(pcm.data()), pcm.size() * sizeof(short));
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
