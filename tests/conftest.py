import sys
from pathlib import Path

# project/ 디렉터리를 Python 경로에 추가
project_root = Path(__file__).parent.parent / "project"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
