#include "targets/placement.h"

#include <cstdlib>

namespace ninfer::targets {

int manual_endpoint_owner_slot() {
    // Read ONCE per process (M4a). The manual loader's shard resolver, its runtime-view condition
    // and the admission printer all consume this same cached value, so the model can never be
    // materialized on one device and viewed or reported on another.
    static const int owner = [] {
        const char* env = std::getenv("NINFER_ENDPOINT_DEVICE");
        if (env == nullptr || (*env != '0' && *env != '1') || *(env + 1) != '\0') { return 0; }
        return *env - '0';
    }();
    return owner;
}

} // namespace ninfer::targets
