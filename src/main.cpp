#include "cartpole.hpp"
#include "dqn.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
    constexpr std::uint32_t seed = 42;
    constexpr int episodes = 500;
    constexpr int warmup_steps = 1000;

    torch::set_num_threads(std::max(1, static_cast<int>(std::thread::hardware_concurrency())));

    const torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << '\n';

    CartPole env(seed);
    DQNAgent agent(device, DQNConfig{}, seed);

    std::vector<float> returns;
    returns.reserve(episodes);
    int total_steps = 0;

    const auto started = std::chrono::steady_clock::now();

    for (int episode = 1; episode <= episodes; ++episode) {
        auto state = env.reset();
        float episode_return = 0.0f;

        for (;;) {
            const int action = agent.select_action(state);
            const auto result = env.step(action);

            agent.remember(Transition{
                state,
                static_cast<int64_t>(action),
                result.reward,
                result.state,
                result.done,
            });

            ++total_steps;
            if (total_steps >= warmup_steps) {
                agent.optimize();
            }

            state = result.state;
            episode_return += result.reward;

            if (result.done) {
                break;
            }
        }

        returns.push_back(episode_return);

        if (episode % 10 == 0) {
            const std::size_t window = std::min<std::size_t>(20, returns.size());
            const float average = std::accumulate(returns.end() - window, returns.end(), 0.0f) /
                                  static_cast<float>(window);

            std::cout << "Episode " << std::setw(3) << episode
                      << " | return " << std::setw(6) << episode_return
                      << " | avg20 " << std::fixed << std::setprecision(1) << std::setw(6) << average
                      << " | epsilon " << std::setprecision(3) << agent.epsilon()
                      << " | replay " << agent.replay_size() << '\n';

            if (average >= 475.0f && window == 20) {
                std::cout << "Solved: average return over the last 20 episodes >= 475.\n";
                break;
            }
        }
    }

    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();

    std::cout << "Training steps: " << total_steps << '\n';
    std::cout << "Elapsed seconds: " << std::fixed << std::setprecision(2) << elapsed << '\n';
    if (elapsed > 0.0) {
        std::cout << "Training throughput: "
                  << static_cast<double>(total_steps) / elapsed
                  << " env steps/s\n";
    }

    return 0;
}
