#!/usr/bin/env bash
set -euo pipefail

qft_repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
qft_env_name="${QFT_CONDA_ENV:-q-transformer}"

# The library uses Python 3.10+; this environment selects Python 3.11.
conda env update -n "$qft_env_name" -f "$qft_repo_dir/reqs/conda.yaml"

# activate conda environment within this script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$qft_env_name"

# Keep installation dependencies in pyproject.toml. The old requirements file
# remains available as a historical experiment snapshot.
python -m pip install -e "$qft_repo_dir[experiments]"
