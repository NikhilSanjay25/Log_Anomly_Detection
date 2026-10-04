"""Create demo inputs in samples/ from real HDFS_v1 sessions.

  samples/hdfs_raw_sample.log      raw HDFS log lines (event sequences of real labelled sessions,
                                   rendered with realistic but synthetic IPs/sizes/timestamps)
  samples/hdfs_sequences.txt       one event sequence per line
  samples/hdfs_traces_sample.csv   Event_traces.csv subset WITH labels (enables evaluation)

Usage: python scripts/make_sample_logs.py [--normal 150 --anomalies 25]
"""
import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from log_agents import config  # noqa: E402

COMP = {"dn": "dfs.DataNode$DataXceiver", "pr": "dfs.DataNode$PacketResponder", "fs": "dfs.FSNamesystem",
        "dn0": "dfs.DataNode", "ds": "dfs.FSDataset", "bs": "dfs.DataBlockScanner", "pm": "dfs.PendingReplicationMonitor"}

RENDER = {  # event -> (level, component, content)  {b}=block {a}{c}=ips {s}=size
    "E1": ("WARN", "fs", "BLOCK* NameSystem.addStoredBlock: Adding an already existing block {b}"),
    "E2": ("INFO", "bs", "Verification succeeded for {b}"),
    "E3": ("INFO", "dn", "{a}:50010 Served block {b} to /{c}"),
    "E4": ("WARN", "dn", "{a}:50010:Got exception while serving {b} to /{c}:"),
    "E5": ("INFO", "dn", "Receiving block {b} src: /{a}:{p} dest: /{c}:50010"),
    "E6": ("INFO", "dn", "Received block {b} src: /{a}:{p} dest: /{c}:50010 of size {s}"),
    "E7": ("INFO", "dn", "writeBlock {b} received exception java.io.IOException: Could not read from stream"),
    "E8": ("INFO", "pr", "PacketResponder 1 for block {b} Interrupted."),
    "E9": ("INFO", "pr", "Received block {b} of size {s} from /{a}"),
    "E10": ("WARN", "pr", "PacketResponder {b} 1 Exception java.io.IOException: Connection reset by peer"),
    "E11": ("INFO", "pr", "PacketResponder 2 for block {b} terminating"),
    "E12": ("INFO", "dn", "{a}:50010:Exception writing block {b} to mirror {c}:50010"),
    "E13": ("INFO", "dn", "Receiving empty packet for block {b}"),
    "E14": ("INFO", "dn", "Exception in receiveBlock for block {b} java.io.IOException: Connection reset by peer"),
    "E15": ("INFO", "dn", "Changing block file offset of block {b} from 0 to 1048576 meta file offset to 8199"),
    "E16": ("INFO", "dn0", "{a}:50010:Transmitted block {b} to /{c}:50010"),
    "E17": ("WARN", "dn0", "{a}:50010:Failed to transfer {b} to {c}:50010 got java.io.IOException: Connection reset by peer"),
    "E18": ("INFO", "dn0", "{a}:50010 Starting thread to transfer block {b} to {c}:50010"),
    "E19": ("INFO", "dn", "Reopen Block {b}"),
    "E20": ("WARN", "ds", "Unexpected error trying to delete block {b}. BlockInfo not found in volumeMap."),
    "E21": ("INFO", "ds", "Deleting block {b} file /mnt/hadoop/dfs/data/current/subdir17/{b}"),
    "E22": ("INFO", "fs", "BLOCK* NameSystem.allocateBlock: /user/root/rand/_temporary/_task_200811101024_0001_m_000{p3}_0/part-00{p3}. {b}"),
    "E23": ("INFO", "fs", "BLOCK* NameSystem.delete: {b} is added to invalidSet of {a}:50010"),
    "E24": ("INFO", "fs", "BLOCK* Removing block {b} from neededReplications as it does not belong to any file."),
    "E25": ("INFO", "fs", "BLOCK* ask {a}:50010 to replicate {b} to datanode(s) {c}:50010"),
    "E26": ("INFO", "fs", "BLOCK* NameSystem.addStoredBlock: blockMap updated: {a}:50010 is added to {b} size {s}"),
    "E27": ("INFO", "fs", "BLOCK* NameSystem.addStoredBlock: Redundant addStoredBlock request received for {b} on {a}:50010 size {s}"),
    "E28": ("INFO", "fs", "BLOCK* NameSystem.addStoredBlock: addStoredBlock request received for {b} on {a}:50010 size {s} But it does not belong to any file."),
    "E29": ("WARN", "pm", "PendingReplicationMonitor timed out block {b}"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", type=int, default=150)
    ap.add_argument("--anomalies", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    traces = pd.read_csv(config.DATASET_DIR / "Event_traces.csv", usecols=["BlockId", "Label", "Features", "Latency"])
    traces["events"] = traces.Features.str.strip("[]").str.split(",")
    traces["n"] = traces.events.map(len)
    traces = traces[traces.n <= 60]
    normal = traces[traces.Label == "Success"].sample(args.normal, random_state=args.seed)
    anom_pool = traces[traces.Label == "Fail"]
    # a diverse anomaly mix: by error-event signature
    anom_pool = anom_pool.assign(sig=anom_pool.events.map(lambda e: tuple(sorted(set(e) - {"E5", "E9", "E11", "E26", "E22", "E3", "E21", "E23"}))))
    anomalies = (anom_pool.groupby("sig", group_keys=False).apply(lambda g: g.head(3))
                 .sample(frac=1, random_state=args.seed).head(args.anomalies))
    chosen = pd.concat([normal, anomalies]).sample(frac=1, random_state=args.seed)

    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    chosen[["BlockId", "Label", "Features", "Latency"]].to_csv(out / "hdfs_traces_sample.csv", index=False)
    with open(out / "hdfs_sequences.txt", "w") as f:
        f.write("# one HDFS block session per line: <block id>: <event ids>\n")
        for r in chosen.itertuples():
            f.write(f"{r.BlockId}: {' '.join(r.events)}\n")

    # raw lines with interleaved timestamps; a handful of DataNodes; one 'flaky' node used more by anomalies
    nodes = [f"10.250.{rng.randint(1, 19)}.{rng.randint(2, 250)}" for _ in range(12)]
    flaky = nodes[0]
    t0 = datetime(2008, 11, 9, 20, 35, 0)
    lines = []
    for r in chosen.itertuples():
        start = t0 + timedelta(seconds=rng.randint(0, 3600))
        is_anom = r.Label == "Fail"
        pipeline = rng.sample(nodes[1:], 3)
        if is_anom and rng.random() < 0.6:
            pipeline[rng.randrange(3)] = flaky
        size = rng.choice([67108864, rng.randint(1000, 67108864)])
        t = start
        for i, ev in enumerate(r.events):
            t += timedelta(seconds=rng.choice([0, 0, 0, 1, 1, 2, 5]))
            if ev in ("E21", "E23") and i and r.events[i - 1] not in ("E21", "E23"):
                t += timedelta(seconds=rng.randint(60, 900))   # deletion happens later
            level, comp, tmpl = RENDER[ev]
            a, c = pipeline[i % 3], pipeline[(i + 1) % 3]
            content = tmpl.format(b=r.BlockId, a=a, c=c, s=size, p=rng.randint(30000, 60000), p3=rng.randint(100, 999))
            lines.append((t, f"{t:%y%m%d %H%M%S} {rng.randint(13, 9999)} {level} {COMP[comp]}: {content}"))
    lines.sort(key=lambda x: x[0])
    raw = [ln for _, ln in lines]
    # a little real-world noise: a duplicated line, an unrelated line, a blank line
    raw.insert(10, raw[9])
    raw.insert(25, "081109 204000 19 INFO dfs.DataNode: Starting DataNode with maxMemory 1014 MB")
    raw.insert(40, "")
    (out / "hdfs_raw_sample.log").write_text("\n".join(raw) + "\n")
    print(f"wrote {len(chosen)} sessions ({len(anomalies)} anomalous) and {len(raw)} raw lines to {out}")


if __name__ == "__main__":
    main()
