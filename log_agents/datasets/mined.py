"""Shared base of the datasets whose event templates are mined from the training logs (BGL, OpenStack).

scripts/train_dataset.py learns the templates with Drain and saves, next to the model artifacts:
  drain_templates.json   the learned templates (DrainParser)
  events.json            one entry per event E<n>: template, component, kind (normal|warning|error), level, counts
At run time the profile loads both and maps every log line to an event id with `event_for`.
"""
import json

import pandas as pd

from .. import config
from ..drain import DrainParser
from .base import ArtifactPaths, DatasetProfile

LEVEL_KIND = {"INFO": "normal", "DEBUG": "normal", "TRACE": "normal", "NOTICE": "normal", "AUDIT": "normal",
              "WARNING": "warning", "WARN": "warning",
              "ERROR": "error", "CRITICAL": "error", "FATAL": "error", "SEVERE": "error", "FAILURE": "error"}


class MinedProfile(DatasetProfile):
    drain_masks = ()
    uses_ips_as_nodes = False

    def __init__(self, directory=None):
        self.dir = directory or (config.ARTIFACTS_ROOT / self.name)
        self._events = None
        self._parser = None
        self._causes = None
        self._reset_derived()

    def _reset_derived(self):
        """Forget what is derived from events.json (subclasses extend this)."""

    # ── artifacts ─────────────────────────────────────────────────────
    @property
    def paths(self):
        return ArtifactPaths.standard(self.dir)

    @property
    def parser_path(self):
        return self.dir / "drain_templates.json"

    @property
    def events_path(self):
        return self.dir / "events.json"

    def install(self, parser, events):
        """Use an in-memory parser and event table (the training script does this before anything is saved)."""
        self._parser, self._events, self._causes = parser, events, None
        self._reset_derived()

    def reload(self):
        self._parser = self._events = self._causes = None
        self._reset_derived()

    @property
    def parser(self):
        if self._parser is None:
            self._parser = DrainParser.load(self.parser_path, self.drain_masks)
        return self._parser

    @property
    def events(self):
        if self._events is None:
            self._events = json.loads(self.events_path.read_text(encoding="utf-8"))
        return self._events

    @property
    def event_info(self):
        return {e: (v["component"], v["kind"], v["template"]) for e, v in self.events.items()}

    @property
    def template_text(self):
        return {e: v["template"] for e, v in self.events.items()}

    def events_matching(self, *keywords, kinds=None):
        """Event ids whose template contains any keyword (case-insensitive), optionally of the given kinds."""
        kws = [k.lower() for k in keywords]
        return {e for e, v in self.events.items()
                if any(k in v["template"].lower() for k in kws) and (kinds is None or v["kind"] in kinds)}

    # ── matching ──────────────────────────────────────────────────────
    def event_for(self, contents):
        """Series of message texts -> Series of event ids (None where no learned template fits)."""
        uniq = contents.unique()
        parser, known = self.parser, self.events
        mapping = {}
        for c in uniq:
            cid = parser.match(c)
            eid = f"E{cid}" if cid is not None else None
            mapping[c] = eid if eid in known else None
        return contents.map(mapping)

    @property
    def root_causes(self):
        if self._causes is None:
            self._causes = self.build_root_causes()
        return self._causes

    def build_root_causes(self):
        raise NotImplementedError


def to_datetime(series, fmt):
    return pd.to_datetime(series, format=fmt, errors="coerce")
