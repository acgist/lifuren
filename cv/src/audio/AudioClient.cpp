#include "lifuren/audio/AudioClient.hpp"

#include "lifuren/audio/AudioModel.hpp"

template<typename M>
std::tuple<bool, std::string> lifuren::AudioClient<M>::pred(const AudioParams& input) {
    // TODO: 实现
    // lifuren::audio::toFile(output)
    return {};
};

std::unique_ptr<lifuren::AudioModelClient> lifuren::getAudioClient(const std::string& client) {
    if(client == lifuren::config::CONFIG_AUDIO_SHIKUANG) {
        return std::make_unique<lifuren::AudioClient<ShikuangModel>>();
    } else {
        return nullptr;
    }
}
