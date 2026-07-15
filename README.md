# data/

Place the dataset CSV here before running `train_model.py`.

**Download:** [Stroke Prediction Dataset by fedesoriano — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

Expected file name:

```
data/healthcare-dataset-stroke-data.csv
```

### Option A — Manual download
1. Go to the Kaggle link above and click **Download**.
2. Unzip it and move `healthcare-dataset-stroke-data.csv` into this `data/` folder.

### Option B — Kaggle API
```bash
pip install kaggle
# place your kaggle.json API key in ~/.kaggle/kaggle.json first
kaggle datasets download -d fedesoriano/stroke-prediction-dataset -p data --unzip
```

This file is intentionally **not committed to the repo** (see `.gitignore`) since Kaggle
datasets shouldn't be redistributed directly — each user should download it with their
own Kaggle account.
