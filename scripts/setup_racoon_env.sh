#!/bin/bash
# Create the "racoon" conda env (Python 3.8, same as the original NFS env) with ReFinED.
# Run once on ai-fe02:  bash scripts/setup_racoon_env.sh
# ReFinED downloads its model files into $REFINED_CACHE_DIR on first use (needs internet).
set -e
ENV_NAME=${ENV_NAME:-racoon}
conda create -y -n "$ENV_NAME" python=3.8
PIP="$(conda run -n "$ENV_NAME" which pip)"
"$PIP" install torch
"$PIP" install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip
"$PIP" install numpy pandas tqdm openai SPARQLWrapper scikit-learn
conda run -n "$ENV_NAME" python -c "import torch, refined; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| refined ok')"
