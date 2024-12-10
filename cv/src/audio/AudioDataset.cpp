#include "lifuren/audio/AudioDataset.hpp"

const static float NORMALIZATION = 32768.0F;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lifuren::dataset::audio::mag_pha_stft(
    torch::Tensor pcm_norm,
    int n_fft,
    int hop_size,
    int win_size,
    float compress_factor
) {
    auto window = torch::hann_window(win_size);
    auto spec = torch::stft(pcm_norm, n_fft, hop_size, win_size, window, true, "reflect", false, std::nullopt, true);
         spec = torch::view_as_real(spec);
    auto mag  = torch::sqrt(spec.pow(2).sum(-1) + (1e-8));
    auto pha  = torch::atan2(spec.index({"...", 1}), spec.index({"...", 0}) + (1e-8));
         mag  = torch::pow(mag, compress_factor);
    auto com  = torch::stack((mag * torch::cos(pha), mag * torch::sin(pha)), -1);
    return std::make_tuple<>(mag, pha, com);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lifuren::dataset::audio::pcm_mag_pha_stft(
    std::vector<short>& pcm,
    float& norm_factor,
    int n_fft,
    int hop_size,
    int win_size,
    float compress_factor
) {
    auto pcm_tensor = torch::zeros({1, static_cast<int>(pcm.size())}, torch::kFloat32);
    float* data = reinterpret_cast<float*>(pcm_tensor.data_ptr());
    std::copy_n(pcm.data(), pcm.size(), data);
    pcm_tensor  = pcm_tensor / NORMALIZATION;
    norm_factor = torch::sqrt(pcm_tensor.sizes()[1] / torch::sum(pcm_tensor.pow(2.0))).template item<float>();
    pcm_tensor  = pcm_tensor * norm_factor;
    return std::move(mag_pha_stft(pcm_tensor, n_fft, hop_size, win_size, compress_factor));
}

torch::Tensor lifuren::dataset::audio::mag_pha_istft(
    torch::Tensor mag,
    torch::Tensor pha,
    int n_fft,
    int hop_size,
    int win_size,
    float compress_factor
) {
    auto window = torch::hann_window(win_size);
         mag = torch::pow(mag, (1.0 / compress_factor));
    auto com = torch::complex(mag * torch::cos(pha), mag * torch::sin(pha));
    return std::move(torch::istft(com, n_fft, hop_size, win_size, window, true));
}

std::vector<short> lifuren::dataset::audio::pcm_mag_pha_istft(
    torch::Tensor mag,
    torch::Tensor pha,
    const float& norm_factor,
    int n_fft,
    int hop_size,
    int win_size,
    float compress_factor
) {
    auto result = mag_pha_istft(mag, pha, n_fft, hop_size, win_size, compress_factor);
         result = result / norm_factor;
         result = result * NORMALIZATION;
    float* data = reinterpret_cast<float*>(result.data_ptr());
    std::vector<short> pcm;
    pcm.resize(result.sizes()[1]);
    std::copy_n(data, pcm.size(), pcm.data());
    return pcm;
}
