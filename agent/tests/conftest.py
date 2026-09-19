import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _no_object_path(monkeypatch):
    """The image-to-3D object path calls FLUX and a local reconstructor; unit tests never should.
    Tests that exercise the routing monkeypatch `orchestrator.object_path_available` themselves."""
    monkeypatch.setenv("COPILOT_OBJECT_PATH", "0")
