"""Guards for the whole suite: nothing here may draw an image, load the 3D worker or call Azure by accident.

config.py loads the developer's .env at import, so a misrouted request in a serve test would otherwise make a
real FLUX call. Tests that need the objects path replace `craftpilot.objects.pipeline.build_object` with a fake.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_network_side_effects(monkeypatch):
    import craftpilot.objects.flow as flow
    import craftpilot.objects.pipeline as pipeline
    import craftpilot.route as route

    def refuse(*args, **kwargs):
        raise AssertionError("build_object must be faked in tests (it would call FLUX and the 3D worker)")

    monkeypatch.setattr(pipeline, "build_object", refuse)
    monkeypatch.setattr(route, "_llm_available", lambda: False)   # the router's LLM tie-break stays off
    monkeypatch.setattr(flow, "unavailable_reason", lambda: None)  # the objects path counts as installed
