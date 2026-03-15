# TCN Baseline (HDFS_v1)

This baseline keeps the original `tcn.py` unchanged and trains a **binary anomaly classifier** on HDFS_v1 BlockId traces.

## Data
Uses:
- `Dataset/HDFS_v1/preprocessed/HDFS.npz` (contains `x_data` and `y_data`)

Each BlockId trace is padded/truncated to `seq_len` and the TCN predicts:
- `0` = normal
- `1` = anomaly

## Run
Notebook:
- `TCN_HDFS_v1.ipynb`

Terminal:
```bash
python train_tcn_hdfs_v1.py --epochs 30 --seq_len 64 --max_traces 50000
```
