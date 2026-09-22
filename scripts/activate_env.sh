# Source from Bash: source scripts/activate_env.sh [train|libero]
# Keeps Python environments local; shared datasets and weights use the requested workspace directories.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Use: source scripts/activate_env.sh [train|libero]" >&2
    exit 1
fi

_contextflow_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-train}" in
    train) _contextflow_venv="$_contextflow_root/.venv" ;;
    libero) _contextflow_venv="$_contextflow_root/examples/libero/.venv" ;;
    *) echo "Expected train or libero" >&2; unset _contextflow_root; return 1 ;;
esac
if [[ ! -f "$_contextflow_venv/bin/activate" ]]; then
    echo "Environment is not installed: $_contextflow_venv" >&2
    unset _contextflow_root _contextflow_venv
    return 1
fi

export UV_CACHE_DIR="$_contextflow_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$_contextflow_root/.cache/python"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-1800}"
export GIT_LFS_SKIP_SMUDGE=1
export HF_HOME="${HF_HOME:-$_contextflow_root/.cache/huggingface}"
# Migrate the defaults from the initial environment setup when re-sourcing it.
if [[ "${LEROBOT_HOME:-}" == "$_contextflow_root/data/lerobot" ]]; then
    unset LEROBOT_HOME
fi
if [[ "${OPENPI_DATA_HOME:-}" == "$_contextflow_root/.cache/openpi" ]]; then
    unset OPENPI_DATA_HOME
fi
export LEROBOT_HOME="${LEROBOT_HOME:-$(dirname -- "$_contextflow_root")/datasets}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$LEROBOT_HOME/.cache/huggingface/datasets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$(dirname -- "$_contextflow_root")/models/openpi}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$_contextflow_root/.cache/libero}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$_contextflow_root/.cache/matplotlib}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_DEFAULT_MATMUL_PRECISION=float32
# Avoid allocating most of every visible GPU when merely importing/testing JAX.
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONPATH="$_contextflow_root/src:$_contextflow_root/third_party/libero:$_contextflow_root/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
source "$_contextflow_venv/bin/activate"
export PATH="$_contextflow_root/.cache/uv-bootstrap/bin:$PATH"
unset _contextflow_root _contextflow_venv
