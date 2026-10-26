"""A compact Drain-style log template miner (He et al., ICWS 2017).

HDFS has 29 hand-written templates (log_agents/hdfs.py). BGL and OpenStack have hundreds of message shapes, so
their templates are mined from the training logs and stored next to the model artifacts:

  * `DrainParser.add(content)` learns templates (offline, scripts/train_dataset.py)
  * `DrainParser.match(content)` maps a line to a learned template id without changing anything (inference);
    it returns None for a line that fits no template, and the pipeline reports those as unmatched.

Lines are first masked (UUIDs, IPs, hex, numbers -> <*>), then routed through a fixed-depth tree keyed on the
token count and the first tokens, and finally compared with the templates in the leaf by token similarity.
"""
import json
import re

WILDCARD = "<*>"

MASKS = [
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), WILDCARD),
    (re.compile(r"(?<![\w.])\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?(?![\w.])"), WILDCARD),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), WILDCARD),
    (re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])"), WILDCARD),
]
_HAS_DIGIT = re.compile(r"\d")
_WORD = re.compile(r"^[A-Za-z]+[,.:;]?$")


class _Node:
    __slots__ = ("children", "clusters")

    def __init__(self):
        self.children = {}
        self.clusters = []          # cluster ids (leaf nodes only)


class DrainParser:
    def __init__(self, depth=4, sim_threshold=0.5, max_children=100, extra_masks=()):
        self.depth = depth
        self.sim_threshold = sim_threshold
        self.max_children = max_children
        self.masks = list(extra_masks) + MASKS
        self.root = {}                      # token count -> _Node
        self.templates = {}                 # cluster id -> list of tokens
        self.sizes = {}                     # cluster id -> number of lines merged into it while mining
        self._cache = {}

    # ── tokenising ────────────────────────────────────────────────────
    def tokenize(self, content):
        for rx, repl in self.masks:
            content = rx.sub(repl, content)
        return content.split()

    # ── tree routing ──────────────────────────────────────────────────
    def _route(self, tokens, create):
        node = self.root.get(len(tokens))
        if node is None:
            if not create:
                return None
            node = self.root[len(tokens)] = _Node()
        for tok in tokens[:self.depth - 2]:
            key = WILDCARD if _HAS_DIGIT.search(tok) else tok
            nxt = node.children.get(key)
            if nxt is None:
                if not create:
                    nxt = node.children.get(WILDCARD)
                    if nxt is None:
                        return None
                else:
                    if key != WILDCARD and len(node.children) >= self.max_children:
                        key = WILDCARD
                    nxt = node.children.setdefault(key, _Node())
            node = nxt
        return node

    @staticmethod
    def _similarity(template, tokens):
        same = wild = 0
        for a, b in zip(template, tokens):
            if a == b:                      # includes <*> against a masked <*>
                same += 1
            elif a == WILDCARD:             # position the template has generalised
                wild += 1
        return same / len(tokens), wild

    @staticmethod
    def _compatible(template, tokens):
        """Two plain words that differ ('spawned' vs 'destroyed') are different events, never one template with a
        variable. Only variable-looking tokens (numbers, ids, paths, hosts) may be generalised."""
        for a, b in zip(template, tokens):
            if a != b and a != WILDCARD and _WORD.match(a) and _WORD.match(b):
                return False
        return True

    def _best(self, node, tokens):
        best, best_sim, best_wild = None, -1.0, -1
        for cid in node.clusters:
            if not self._compatible(self.templates[cid], tokens):
                continue
            sim, wild = self._similarity(self.templates[cid], tokens)
            if sim > best_sim or (sim == best_sim and wild > best_wild):
                best, best_sim, best_wild = cid, sim, wild
        return best, best_sim

    # ── learning ──────────────────────────────────────────────────────
    def add(self, content):
        """Learn from one line; returns its cluster id (templates may be generalised by the line)."""
        tokens = self.tokenize(content)
        if not tokens:
            return None
        node = self._route(tokens, create=True)
        cid, sim = self._best(node, tokens)
        if cid is not None and sim >= self.sim_threshold:
            tpl = self.templates[cid]
            self.templates[cid] = [a if a == b else WILDCARD for a, b in zip(tpl, tokens)]
            self.sizes[cid] += 1
            return cid
        cid = len(self.templates) + 1
        self.templates[cid] = list(tokens)
        self.sizes[cid] = 1
        node.clusters.append(cid)
        return cid

    # ── inference ─────────────────────────────────────────────────────
    def match(self, content):
        """Cluster id of the learned template that fits `content`, or None. Never changes the parser."""
        tokens = self.tokenize(content)
        key = " ".join(tokens)
        if key in self._cache:
            return self._cache[key]
        cid = None
        if tokens:
            node = self._route(tokens, create=False)
            if node is not None:
                best, sim = self._best(node, tokens)
                if best is not None and sim >= self.sim_threshold:
                    cid = best
        self._cache[key] = cid
        return cid

    def template_text(self, cid):
        return " ".join(self.templates[cid])

    # ── persistence ───────────────────────────────────────────────────
    def to_dict(self):
        return {"depth": self.depth, "sim_threshold": self.sim_threshold, "max_children": self.max_children,
                "templates": {str(c): t for c, t in self.templates.items()},
                "sizes": {str(c): n for c, n in self.sizes.items()}}

    @classmethod
    def from_dict(cls, d, extra_masks=()):
        p = cls(d["depth"], d["sim_threshold"], d["max_children"], extra_masks)
        for cid, tokens in d["templates"].items():
            cid = int(cid)
            p.templates[cid] = list(tokens)
            p.sizes[cid] = int(d.get("sizes", {}).get(str(cid), 0))
            p._route(tokens, create=True).clusters.append(cid)
        return p

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=0)

    @classmethod
    def load(cls, path, extra_masks=()):
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f), extra_masks)
