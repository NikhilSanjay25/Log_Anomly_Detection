"""Test configuration: the default run is rule-based (no language model), so the suite is fast and needs no GPU.
Tests of the investigator pass `agent_model` explicitly (usually with a scripted fake model)."""
import os

os.environ.setdefault("LAD_AGENT_MODEL", "rules")
