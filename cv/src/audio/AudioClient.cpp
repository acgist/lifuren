#include "lifuren/audio/Audio.hpp"

#include "lifuren/audio/AudioModel.hpp"

template<typename M>
std::tuple<bool, std::string> lifuren::audio::AudioClient<M>::pred(const AudioParams& input) {
    // TODO: 实现
    // lifuren::audio::toFile(output)
    return {};
};

std::unique_ptr<lifuren::audio::AudioModelClient> lifuren::audio::getAudioClient(const std::string& model) {
    if(model == lifuren::config::CONFIG_AUDIO_SHIKUANG) {
        return std::make_unique<lifuren::audio::AudioClient<ShikuangModel>>();
    } else {
        return nullptr;
    }
}
