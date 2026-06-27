import sys
import traceback
import os
from pathlib import Path

VENDOR = Path(__file__).resolve().parent / "_vendor"
sys.path.append(str(VENDOR))
YAML_VENDOR = VENDOR / "yaml"
if YAML_VENDOR.exists():
    sys.path.append(str(YAML_VENDOR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_DISABLE_GENERATION_IMPORT", "1")

print("step=import")
from transformers.models.auto.modeling_auto import AutoModel  # noqa: E402
from transformers.models.auto.tokenization_auto import AutoTokenizer  # noqa: E402

print("step=tokenizer-start")
try:
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5", trust_remote_code=True)
    print("step=tokenizer-ok", type(tokenizer).__name__)
    print("step=model-start")
    model = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5", trust_remote_code=True)
    print("step=model-ok", type(model).__name__)
except Exception as exc:
    print("step=error", type(exc).__name__)
    traceback.print_exc()
