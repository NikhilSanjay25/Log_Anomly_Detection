"""Create demo inputs in samples/ from the real BGL and OpenStack logs.

  samples/bgl_raw_sample.log            raw BGL RAS lines: 45 whole 100-message windows from the held-out (last 20%)
                                        part of the log, 15 of them containing alerts (chosen at random, seeded)
  samples/openstack_raw_sample.log      raw OpenStack nova lines: every instance-tagged line of 40 normal instances
                                        and the 2 labelled-anomalous instances, plus untagged lines (API polling...)
  samples/openstack_raw_sample.labels.json   which instances of that sample are the labelled anomalies

The BGL windows come from the part of the log the BGL model was not trained on. The OpenStack model is trained on
all instances (there are only 4 labelled anomalies), so its sample is not held out.

Usage: python scripts/make_dataset_samples.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from log_agents import config  # noqa: E402
from log_agents.datasets import get_profile  # noqa: E402

SAMPLES = ROOT / "samples"
SEED = 7


def bgl_sample():
    profile = get_profile("bgl")
    lines = (config.BGL_DATASET_DIR / "BGL.log").read_text(encoding="utf-8", errors="replace").splitlines()
    tail = "\n".join(lines[int(len(lines) * 0.82):])          # safely inside the held-out last 20%
    bronze, _ = profile.clean(tail, "bgl_raw")
    full = bronze.groupby("block_id").size()
    full = full[full == config.BGL_WINDOW].index
    label = bronze.groupby("block_id").true_label.max()
    anomalous, normal = [w for w in full if label[w] == 1], [w for w in full if label[w] == 0]
    rng = random.Random(SEED)
    chosen = sorted(rng.sample(anomalous, 15) + rng.sample(normal, 30))
    out = bronze[bronze.block_id.isin(chosen)].sort_values("line_no")
    (SAMPLES / "bgl_raw_sample.log").write_text("\n".join(out.raw) + "\n", encoding="utf-8")
    print(f"bgl_raw_sample.log: {len(chosen)} windows ({len(out):,} lines), 15 with alerts")


def openstack_sample():
    profile = get_profile("openstack")
    src = config.OPENSTACK_DATASET_DIR
    bad = [ln.strip() for ln in (src / "anomaly_labels.txt").read_text().splitlines()
           if len(ln.strip()) == 36 and ln.count("-") == 4]
    # two anomalies and a spread of normal instances, each kept whole
    abnormal = (src / "openstack_abnormal.log").read_text(encoding="utf-8", errors="replace").splitlines()
    normal2 = (src / "openstack_normal2.log").read_text(encoding="utf-8", errors="replace").splitlines()
    rng = random.Random(SEED)
    sample_bad = bad[:2]
    bronze_n, _ = profile.clean("\n".join(normal2), "openstack_raw")
    normal_ids = sorted(bronze_n.block_id.dropna().unique())
    sample_ids = set(rng.sample(normal_ids, 40)) | set(sample_bad)
    keep = []
    for name, lines in (("abnormal", abnormal), ("normal2", normal2)):
        bronze, _ = profile.clean("\n".join(lines), "openstack_raw")
        tagged = bronze[bronze.block_id.isin(sample_ids)]
        untagged = bronze[bronze.block_id.isna()].sample(80, random_state=SEED)
        keep.append(pd_concat(tagged, untagged))
    out = pd_concat(*keep).sort_values("timestamp", kind="stable")
    (SAMPLES / "openstack_raw_sample.log").write_text("\n".join(out.raw) + "\n", encoding="utf-8")
    (SAMPLES / "openstack_raw_sample.labels.json").write_text(
        json.dumps({"anomalous_instances": sample_bad, "total_instances": len(sample_ids)}, indent=1))
    print(f"openstack_raw_sample.log: {len(sample_ids)} instances ({len(out):,} lines), anomalies {sample_bad}")


def pd_concat(*frames):
    import pandas as pd
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    SAMPLES.mkdir(exist_ok=True)
    bgl_sample()
    openstack_sample()
