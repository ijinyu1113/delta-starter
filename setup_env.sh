#!/bin/bash
# Run once on a Delta login node, from inside the repo:
#     bash setup_env.sh
#
# Creates .venv/ : a Python environment that reuses the PyTorch from Delta's
# pytorch-conda module and adds the libraries in requirements.txt.
# Rerun it any time you add a line to requirements.txt.
set -e

module reset
module load pytorch-conda

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
echo "Environment ready in .venv/"
