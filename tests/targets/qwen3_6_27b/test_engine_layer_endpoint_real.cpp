// M4b: endpoint execution under `--split-mode layer` for BOTH endpoint owners, through every
// runtime path the CLI cannot reach in one process. Run twice -- once with NINFER_ENDPOINT_DEVICE
// unset (owner slot 0, the default) and once with NINFER_ENDPOINT_DEVICE=1 -- and compare the
// printed ENDPOINT_TOKENS lines: greedy token identity across owners is the correctness gate.
//
// Paths exercised in-process:
//   1. prefill (chunked; last-column projection)            -> baseline generation
//   2. ordinary decode (normalized hidden + logits return)  -> baseline generation
//   3. zero-suffix prefix reuse -> `sample_from_hidden` (retained post-norm tail hidden, head only)
//      -- this path is unreachable from the CLI (one request per process) and from ninfer-serve
//      (no layer split), so this test is its only live coverage.
//   4. repeated execution (a second independent generation must reproduce the first)
//
// Gates: NINFER_QWEN3_8_27B_WEIGHTS (artifact path), two CUDA devices; NINFER_LAYER_SPLIT_BOUNDARY
// selects the boundary (default 57). Skips with 77 otherwise.
#include <ninfer/engine.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

ninfer::EngineOptions options(const char* artifact) {
    ninfer::EngineOptions out;
    out.artifact_path        = artifact;
    out.max_context          = 8192;
    out.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(8192);
    out.prefill_chunk        = 1024;
    out.kv_cache             = ninfer::KvCacheStorage::Int8Group64;
    out.use_cuda_graph       = false;
    out.devices              = {0, 1};
    out.split_mode           = ninfer::SplitMode::Layer;
    out.layer_split_boundary = 57;
    if (const char* boundary = std::getenv("NINFER_LAYER_SPLIT_BOUNDARY");
        boundary != nullptr && *boundary != '\0') {
        out.layer_split_boundary = std::atoi(boundary);
    }
    return out;
}

ninfer::RequestOptions request(std::uint32_t tokens, bool reuse) {
    ninfer::RequestOptions out;
    out.execution.requested_output_tokens = tokens;
    out.execution.sampling.temperature    = 0.0F;
    out.execution.allow_prefix_reuse      = reuse;
    out.stop.include_model_defaults       = false;
    return out;
}

std::string join(const std::vector<ninfer::TokenId>& ids) {
    std::string out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) { out += ' '; }
        out += std::to_string(ids[i]);
    }
    return out;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_WEIGHTS is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: layer split endpoint test needs two CUDA devices\n";
        return 77;
    }
    const char* owner_env = std::getenv("NINFER_ENDPOINT_DEVICE");
    std::cout << "ENDPOINT_OWNER_ENV " << (owner_env == nullptr ? "(unset)" : owner_env) << '\n';
    try {
        ninfer::Engine engine(options(artifact));
        // A prompt spanning more than one 1024-token prefill chunk, so the norm-only chunk path
        // (head_columns == 0) runs before the final projected chunk.
        // Deterministic pseudo-random ids (not a repeated token) so the greedy continuation is a
        // non-degenerate sequence: an owner mismatch or a stale logits buffer shows up as a token
        // difference rather than hiding behind a constant output.
        std::vector<ninfer::TokenId> prompt;
        prompt.reserve(1502);
        for (std::uint32_t i = 0; i < 1500; ++i) {
            prompt.push_back(static_cast<ninfer::TokenId>(100 + (i * 7919U) % 20000U));
        }
        prompt.push_back(9707);
        prompt.push_back(11);

        // 1+2. prefill + decode: eight greedy tokens.
        const ninfer::GenerationResult baseline =
            engine.generate(engine.prepare_tokens(prompt), request(8, false));
        if (baseline.generated_token_ids.size() != 8) {
            std::cerr << "baseline did not generate eight tokens\n";
            return 1;
        }
        std::cout << "ENDPOINT_TOKENS baseline " << join(baseline.generated_token_ids) << '\n';

        // 4. repeated execution: no stale staging, same output.
        const ninfer::GenerationResult repeat =
            engine.generate(engine.prepare_tokens(prompt), request(8, false));
        if (repeat.generated_token_ids != baseline.generated_token_ids) {
            std::cerr << "repeated generation diverged: " << join(repeat.generated_token_ids) << '\n';
            return 1;
        }

        // 3. zero-suffix prefix reuse -> sample_from_hidden on the retained tail hidden.
        std::vector<ninfer::TokenId> exact_frontier = prompt;
        exact_frontier.insert(exact_frontier.end(), baseline.generated_token_ids.begin(),
                              baseline.generated_token_ids.end() - 1);
        const ninfer::GenerationResult reused =
            engine.generate(engine.prepare_tokens(exact_frontier), request(2, true));
        if (reused.reused_prompt_tokens != exact_frontier.size()) {
            std::cerr << "zero-suffix reuse count is " << reused.reused_prompt_tokens
                      << ", expected " << exact_frontier.size() << '\n';
            return 1;
        }
        if (reused.generated_token_ids.size() != 2 ||
            reused.generated_token_ids[0] != baseline.generated_token_ids.back()) {
            std::cerr << "zero-suffix reuse did not resume from the retained tail hidden: "
                      << join(reused.generated_token_ids) << '\n';
            return 1;
        }
        std::cout << "ENDPOINT_TOKENS zero_suffix " << join(reused.generated_token_ids) << '\n';
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
