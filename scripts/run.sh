#!/usr/bin/env bash
set -euo pipefail
python -m streamlit run app/app.py --server.port 8501 --server.address 0.0.0.0
