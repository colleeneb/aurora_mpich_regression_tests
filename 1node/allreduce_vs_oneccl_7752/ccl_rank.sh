#!/usr/bin/env bash

# https://github.com/uxlfoundation/oneCCL/blob/66499938b7a8b615e26361c52900e7aec306ce50/src/common/global/global.cpp#L187
export CCL_PROCESS_LAUNCHER=none
export CCL_LOCAL_RANK=$PALS_LOCAL_RANKID
export CCL_LOCAL_SIZE=$PALS_LOCAL_SIZE

exec "$@"
