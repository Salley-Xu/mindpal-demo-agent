This directory stores offline runtime dependencies used by `agent_test_data`.

- `vendor/`
  - Vendored Python packages loaded by `agent_test_data/faiss_utils.py` in restricted environments.
- `wheelhouse/`
  - Cached wheel files kept for rebuilding or supplementing the vendored runtime offline.
- `models/`
  - Local Hugging Face model directories used by the FAISS embedding pipeline.

These files are intentionally kept outside `agent_test_data/` so the test data directory stays focused on datasets, scripts, and evaluation docs.
