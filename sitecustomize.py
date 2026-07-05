import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache"
MATPLOTLIB_CACHE = CACHE_ROOT / "matplotlib"
FONTCONFIG_CACHE = CACHE_ROOT / "fontconfig"

for path in (CACHE_ROOT, MATPLOTLIB_CACHE, FONTCONFIG_CACHE):
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE))
os.environ.setdefault("FONTCONFIG_PATH", str(FONTCONFIG_CACHE))
