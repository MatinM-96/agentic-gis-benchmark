from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backend.mcp.server.registry import mcp
from backend.mcp.server.discovery_tools import *
from backend.mcp.server.step_tools import *

if __name__ == "__main__":
    mcp.run()