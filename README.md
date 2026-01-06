# AIVantage

AIVantage is a research prototype for learning dense market representations from OHLCV data. It converts technical indicators into compact image-like windows, trains a masked convolutional autoencoder, and uses the learned embeddings for downstream market classification.

## Pipeline

1. `data_main.py` converts OHLCV CSV files into `15 x 12` indicator windows.
2. `labeller_corr.py` assigns classes from forward price movement.
3. `mae.py` trains a masked autoencoder to learn dense representations.
4. `transformer_class.py` trains a transformer classifier on sequences of those representations.
5. `xgb.py` and `knn.py` provide optional downstream evaluation baselines. `train_cnn.py` provides a direct CNN baseline.

## Requirements

- Python 3.10 or newer
- An NVIDIA GPU with a CUDA-enabled PyTorch installation for the training scripts
- Historical OHLCV CSV files with `timestamp`, `open`, `high`, `low`, `close`, and `volume` columns

Install the Python dependencies in a virtual environment:

```bash
git clone https://github.com/ishnbl/AIVantage.git
cd AIVantage
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset used during development is available in this [Google Drive folder](https://drive.google.com/drive/folders/1Q15OlMBxMhhYnIsxWgxhCzMdIUyfYKdL?usp=sharing). Place the CSV files in a directory named `raw_data` at the repository root.

## Run the pipeline

Generate indicator windows:

```bash
python data_main.py
```

The preprocessing script writes to `images2`, while the labelling script reads from `images`. Rename the generated directory before continuing, or update the corresponding path constants in the scripts:

```bash
mv images2 images
python labeller_corr.py
```

Train the masked autoencoder and transformer classifier:

```bash
python mae.py
python transformer_class.py
```

Run the optional baselines after their expected checkpoints have been generated:

```bash
python train_cnn.py
python xgb.py
python knn.py
```

The scripts define their dataset, checkpoint, and output paths near the top of each file. Update those constants when using a different directory layout.

## Repository structure

| File | Purpose |
| --- | --- |
| `data_main.py` | Builds technical-indicator windows from raw OHLCV data. |
| `labeller_corr.py` | Relabels windows using forward returns. |
| `mae.py` | Trains the masked convolutional autoencoder. |
| `transformer_class.py` | Trains a transformer over sequences of learned representations. |
| `train_cnn.py` | Trains a direct CNN classification baseline. |
| `generate_vector.py` | Exports encoder representations to an NPZ database. |
| `xgb.py` | Trains and evaluates an XGBoost classifier. |
| `knn.py` | Evaluates a nearest-neighbour classifier. |

## Notes

This repository is an experimental research codebase. It is not financial advice or a production trading system.
