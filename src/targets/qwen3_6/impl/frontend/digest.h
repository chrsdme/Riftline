#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6::frontend_internal {

using Sha256Digest = std::array<std::uint8_t, 32>;

// Incremental SHA256 context: feed data via update() in any number of calls, then finish() once.
// The whole-buffer sha256() overloads below are implemented through this context so there is
// exactly one SHA256 algorithm in this translation unit. No checkpoint callback here -- the
// existing offset-based checkpoint cadence (see sha256(span, checkpoint) below) is tied to a
// single contiguous span's local offset and is preserved only on that whole-buffer overload;
// streaming callers (e.g. sha256_hex_of_file()) do not need mid-hash checkpoints.
class Sha256Context {
public:
    Sha256Context();

    void update(std::span<const std::uint8_t> data);

    [[nodiscard]] Sha256Digest finish();

private:
    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> buffer_{};
    std::size_t buffer_used_ = 0;
    std::uint64_t total_bytes_ = 0;
};

[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input);
[[nodiscard]] Sha256Digest sha256(std::span<const std::uint8_t> input,
                                  const std::function<void()>& checkpoint);
[[nodiscard]] Sha256Digest sha256(std::string_view input);
[[nodiscard]] std::string sha256_hex(const Sha256Digest& digest);

} // namespace ninfer::targets::qwen3_6::frontend_internal
