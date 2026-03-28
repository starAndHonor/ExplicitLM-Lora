#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

exp_load_env() {
    local env_file="${PROJECT_ROOT}/.env"
    if [ -f "${env_file}" ]; then
        set -a
        # shellcheck disable=SC1090
        source "${env_file}"
        set +a
    fi
}

exp_first_gpu() {
    local gpu_ids="$1"
    echo "${gpu_ids%%,*}"
}

exp_ckpt_tag() {
    local ckpt="$1"
    local normalized="${ckpt%/}"
    local base
    local parent
    base="$(basename "${normalized}")"
    parent="$(basename "$(dirname "${normalized}")")"
    case "${base}" in
        phase2_best|phase3_best|phase2_epoch*|phase3_epoch*)
            if [ -n "${parent}" ] && [ "${parent}" != "." ] && [ "${parent}" != "checkpoints" ]; then
                echo "${parent}_${base}"
            else
                echo "${base}"
            fi
            ;;
        *)
            echo "${base}"
            ;;
    esac
}

exp_override_args() {
    local enc_mode="$1"
    if [ "${enc_mode}" = "qwen3" ]; then
        echo "--override model.knowledge_encoder_mode=qwen3"
    fi
}

exp_print_cmd() {
    printf '[Experiment] Command:'
    for arg in "$@"; do
        printf ' %q' "${arg}"
    done
    printf '\n'
}
