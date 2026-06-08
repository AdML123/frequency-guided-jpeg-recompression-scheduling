from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SRC_ROOT_TEXT = str(SRC_ROOT)
REPO_ROOT_TEXT = str(REPO_ROOT)

sys.path = [path for path in sys.path if path not in {SRC_ROOT_TEXT, REPO_ROOT_TEXT}]
sys.path.insert(0, REPO_ROOT_TEXT)
sys.path.insert(0, SRC_ROOT_TEXT)

for module_name, module in list(sys.modules.items()):
    if module_name != "jpeg_defense" and not module_name.startswith("jpeg_defense."):
        continue
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        continue
    if not Path(module_file).resolve().is_relative_to(SRC_ROOT):
        del sys.modules[module_name]
