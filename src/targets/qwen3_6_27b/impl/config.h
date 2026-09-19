#pragma once

#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/hybrid_topology.h>
#include <ninfer/targets/qwen3_6/vision.h>

#include "targets/placement.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6_27b::detail {

// Default layer-split device boundary: layers [0, boundary) live on device 0,
// [boundary, TextConfig::layers) on device 1. C2: the boundary is now a runtime value threaded in
// from EngineOptions::layer_split_boundary (see resolve_layer_split_boundary in package.cpp), not
// a compile-time constant -- this is only the default applied when the caller omits it. Kept here
// (relocated from impl/load/bindings.h, B2 Stage 3) so both the loader (bindings.h/.cpp) and the
// generic runtime's per-device state sizing (layouts_impl.h, via Variant::TextConfig) can derive
// owner-local layer counts from the SAME resolved value -- no independent hardcodes.
inline constexpr std::size_t kDefaultLayerSplitBoundary = 56;

[[nodiscard]] constexpr int layer_owner_for_layer(std::size_t layer, std::size_t boundary) {
    return layer < boundary ? 0 : 1;
}


struct TextConfig {
    static constexpr int hidden       = 5120;
    static constexpr int layers       = 64;
    static constexpr int intermediate = 17408;

    // The output matrix is padded for the selected kernels. Only token IDs in
    // [0, token_domain) are tokenizer-addressable and valid sampling results.
    static constexpr int output_rows  = 248320;
    static constexpr int token_domain = static_cast<int>(qwen3_6::kTokenDomain);

    static constexpr int gdn_conv_kernel      = 4;
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    static constexpr int gdn_key_heads        = 16;
    static constexpr int gdn_key_head_dim     = 128;
    static constexpr int gdn_value_heads      = 48;
    static constexpr int gdn_value_head_dim   = 128;

    static constexpr int query_heads = 24;
    static constexpr int kv_heads    = 4;
    static constexpr int head_dim    = 256;
    static constexpr int rotary_dim  = 64;

    static constexpr int full_attention_interval = qwen3_6::kHybridAttentionInterval;
    static constexpr float rms_epsilon           = 1.0e-6F;
    static constexpr float rope_theta            = 1.0e7F;

    static constexpr int key_dim               = gdn_key_heads * gdn_key_head_dim;
    static constexpr int value_dim             = gdn_value_heads * gdn_value_head_dim;
    static constexpr int convolution_dim       = 2 * key_dim + value_dim;
    static constexpr int query_size            = query_heads * head_dim;
    static constexpr int kv_size               = kv_heads * head_dim;
    static constexpr int query_projection_rows = 2 * query_size;

    static constexpr int mtp_layers               = 1;
    static constexpr int mtp_input_rows           = 2 * hidden;
    static constexpr int mtp_attention_input_rows = 2 * query_size + 2 * kv_size;
    static constexpr int mtp_mlp_gate_up_rows     = 2 * intermediate;

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        return qwen3_6::is_full_attention_layer(layer);
    }

    [[nodiscard]] static constexpr int full_attention_layers() {
        return qwen3_6::full_attention_layers(layers);
    }

    [[nodiscard]] static constexpr int gdn_layers() { return qwen3_6::gdn_layers(layers); }

    [[nodiscard]] static constexpr int full_attention_index(int layer) {
        return qwen3_6::full_attention_index(layer);
    }

    [[nodiscard]] static constexpr int gdn_index(int layer) { return qwen3_6::gdn_index(layer); }

    // Count of this family's (full-attention or GDN) layers owned by `owner` under the layer
    // split. Used to size per-device KV/GDN state to only the layers a device actually executes,
    // and to remap a layer's global family-local index (full_idx/gdn_idx) down to an owner-local
    // one at the KV/GDN state sites. Owner 0 always owns a contiguous prefix, so its layers keep
    // their global family-local index unchanged (offset 0); only owner 1's indices need shifting.
    // `boundary` is a runtime layer-split boundary (see kDefaultLayerSplitBoundary); these remain
    // constexpr functions purely so the DEFAULT-boundary static_asserts below stay compile-time
    // sanity checks, but they are called with runtime values from bindings.cpp/layouts_impl.h.
    [[nodiscard]] static constexpr int owned_full_attention_layers(int owner,
                                                                    std::size_t boundary) {
        int count = 0;
        for (int layer = 0; layer < layers; ++layer) {
            if (is_full_attention(layer) &&
                layer_owner_for_layer(static_cast<std::size_t>(layer), boundary) == owner) {
                ++count;
            }
        }
        return count;
    }

    [[nodiscard]] static constexpr int owned_gdn_layers(int owner, std::size_t boundary) {
        int count = 0;
        for (int layer = 0; layer < layers; ++layer) {
            if (!is_full_attention(layer) &&
                layer_owner_for_layer(static_cast<std::size_t>(layer), boundary) == owner) {
                ++count;
            }
        }
        return count;
    }
};

static_assert(TextConfig::full_attention_layers() == 16);
static_assert(TextConfig::gdn_layers() == 48);
// Every layer belongs to exactly one owner under the layer split -- the per-device sizing in
// layouts_impl.h relies on these two counts partitioning the full layer set with no overlap or
// gap. Sum invariants rather than literal counts so a test-only boundary change (e.g. 60/4
// admission check) does not need this assertion edited to pass. Evaluated at the DEFAULT boundary
// only, as compile-time sanity; runtime boundaries are covered by ninfer_qwen3_6_27b_shard_map_test.
static_assert(TextConfig::owned_full_attention_layers(0, kDefaultLayerSplitBoundary) +
                 TextConfig::owned_full_attention_layers(1, kDefaultLayerSplitBoundary) ==
             TextConfig::full_attention_layers());
static_assert(TextConfig::owned_gdn_layers(0, kDefaultLayerSplitBoundary) +
                 TextConfig::owned_gdn_layers(1, kDefaultLayerSplitBoundary) ==
             TextConfig::gdn_layers());
static_assert(TextConfig::owned_full_attention_layers(0, kDefaultLayerSplitBoundary) == 14);
static_assert(TextConfig::owned_gdn_layers(0, kDefaultLayerSplitBoundary) == 42);

// M5.0: the ONE placement authority. Both the real loader (shard resolver + runtime view) and the
// analytical placement planner derive every ownership decision from a PlacementSpec through these
// two functions -- never from the environment, never from a second table.
[[nodiscard]] constexpr int owner_for_layer(const PlacementSpec& spec, std::size_t layer) {
    return layer_owner_for_layer(layer, static_cast<std::size_t>(spec.boundary));
}

// Owner of an artifact object under layer split. The output endpoint group (`text/final_norm`,
// `text/output_head`) follows `spec.endpoint_owner_slot`; `text/layers/N/...` follows the boundary;
// everything else (token embedding, frontend resources, MTP/vision objects that layer split never
// materializes) stays on slot 0. Pure: no getenv, no statics.
[[nodiscard]] inline int owner_for_object(const PlacementSpec& spec, std::string_view object) {
    constexpr std::string_view prefix = "text/layers/";
    if (object == "text/final_norm" || object == "text/output_head") {
        return spec.endpoint_owner_slot;
    }
    if (!object.starts_with(prefix)) { return 0; }
    std::uint64_t layer = 0;
    std::size_t pos     = prefix.size();
    if (pos >= object.size() || object[pos] < '0' || object[pos] > '9') { return 0; }
    while (pos < object.size() && object[pos] >= '0' && object[pos] <= '9') {
        layer = layer * 10 + static_cast<unsigned>(object[pos] - '0');
        ++pos;
    }
    if (pos >= object.size() || object[pos] != '/' || layer >= static_cast<std::uint64_t>(TextConfig::layers)) { return 0; }
    return owner_for_layer(spec, layer);
}

// Resolves the CLI/EngineOptions `--layer-split` request into the canonical runtime boundary: 0
// means "use the default", otherwise the requested value is range-checked against this target's
// layer count. This is the ONLY place layer_split_boundary is resolved -- callers (Package::
// resolve_options) must route every use of the boundary through this function's result rather than
// re-deriving it, so the loader and the sequence planner never see different values for the same
// request.
[[nodiscard]] inline int resolve_layer_split_boundary(int requested) {
    if (requested == 0) { return static_cast<int>(kDefaultLayerSplitBoundary); }
    if (requested <= 0 || requested >= TextConfig::layers) {
        throw std::invalid_argument(
            "--layer-split " + std::to_string(requested) + " is out of range: must be in [1, " +
            std::to_string(TextConfig::layers - 1) + "] for a " +
            std::to_string(TextConfig::layers) + "-layer model");
    }
    return requested;
}

struct VisionConfig : qwen3_6::VisionBackboneConfig {
    static constexpr int output_hidden = TextConfig::hidden;
};

struct DFlashConfig {
    static constexpr bool supported     = false;
    static constexpr int local_layers   = 0;
    static constexpr int local_capacity = 0;
    static constexpr int kv_heads       = 0;
    static constexpr int head_dim       = 0;
    static constexpr int feature_rows   = 0;
    static constexpr int hidden         = 0;
    static constexpr int intermediate   = 0;
    static constexpr int query_size     = 0;
    static constexpr int kv_size        = 0;
};

inline constexpr float kAttentionScale                   = 0.0625F;
inline constexpr float kGdnScale                         = 0.08838834764831845F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
inline constexpr std::uint32_t kNativeContext            = 262144;

} // namespace ninfer::targets::qwen3_6_27b::detail
