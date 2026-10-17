# VK-LSVD Clip Recommendation: Collaborative Filtering Benchmarks

A Python recommender-system project for ranking video clips for users using interaction data from the [VK-LSVD dataset](https://huggingface.co/datasets/deepvk/VK-LSVD).

The project compares several collaborative-filtering approaches against a simple popularity baseline and exports ranked recommendations to CSV. It was developed for an educational assignment where **MAP@10** is the primary metric.

> **Status:** This repository contains the implementation and evaluation pipeline. Benchmark scores must be generated on your own validation/test split; no performance numbers are claimed here.

## Features

- Compare multiple recommendation approaches in one run.
- Evaluate recommendations using MAP@K, Precision@K, Recall@K, NDCG@K, and catalog coverage.
- Exclude items already present in a user's training history from recommendations.
- Fall back to popularity-based recommendations for users without a usable positive profile.
- Save ranked recommendations and a separate metrics table as CSV files.
- Tune key model parameters from the command line.

## Implemented methods

| Method | Description | Main parameters |
|---|---|---|
| **Top Popular** | Popularity baseline based on the number of interaction rows per item, following the original baseline approach. | — |
| **ItemKNN** | Item-item collaborative filtering using cosine similarity between binary user-item interaction vectors. | `--knn_max_items`, `--knn_neighbors` |
| **EASE** | Regularized linear item-item model with a closed-form solution. The inverse requires a dense item-item matrix, so training is limited to the most popular items. | `--ease_max_items`, `--ease_reg` |
| **SLIM** | Sparse linear item-item model. It fits an ElasticNet regression for each target item and prunes the learned coefficients. | `--slim_max_items`, `--slim_alpha`, `--slim_l1_ratio`, `--slim_neighbors` |
| **ALS / iALS** | Alternating Least Squares for implicit feedback, implemented using `implicit.als.AlternatingLeastSquares`. | `--factors`, `--iterations`, `--regularization`, `--alpha` |
| **TruncatedSVD** | Low-rank factorization baseline from scikit-learn. | `--factors` |

### How interaction signals are used

The script constructs a heuristic confidence weight from watch time, likes, shares, bookmarks, author-page clicks, and comment opens. Rows marked as dislikes receive zero positive weight. ALS uses this weighted interaction matrix. TruncatedSVD also uses the weighted matrix. ItemKNN, EASE, and SLIM use a binary matrix of positive interactions, so the relative values of watch time and positive action weights do not directly change their training targets.

The weights are engineering choices, not learned parameters. They should be treated as tunable assumptions and evaluated on a validation split.

## Repository structure

```text
vk-lsvd-recommender/
├── README.md
├── requirements.txt
├── .gitignore
└── solution_cf.py
```

The dataset files are intentionally not included. Download or prepare them separately and keep large data files out of the Git repository unless there is a specific reason to distribute them.

## Dataset format

The script expects Parquet files containing at least these columns:

- `user_id`: user identifier
- `item_id`: video/clip identifier

Additional supported columns include:

- `timespent`
- `like`
- `dislike`
- `share`
- `bookmark`
- `click_on_author`
- `open_comments`

The training file is used to build the interaction matrix and the item catalog. For evaluation, the hold-out file must include both `user_id` and `item_id`.

## Installation

Python 3.10 or newer is recommended. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the pinned packages:

```bash
pip install -r requirements.txt
```

The `implicit` package is required for ALS/iALS. If it cannot be installed in your environment, the script can still compare other methods when ALS fails to train, but the ALS result will be absent.

## Quick start

### 1. Compare all models on a labelled hold-out split

```bash
python solution_cf.py \
  --input_path train.parquet \
  --test_path validation.parquet \
  --output_path submission.csv
```

On Windows PowerShell, the same command can be entered on one line:

```powershell
python solution_cf.py --input_path train.parquet --test_path validation.parquet --output_path submission.csv
```

When a valid hold-out file is supplied, the script attempts to train and evaluate Top Popular, ItemKNN, EASE, SLIM, ALS, and TruncatedSVD. A model that fails to train is reported and skipped. The metric table is sorted by MAP@K and written to `submission_metrics.csv` by default.

### 2. Generate recommendations with a chosen model

For example, to generate recommendations with EASE:

```bash
python solution_cf.py --input_path train.parquet --output_path submission.csv --model ease
```

To use SLIM:

```bash
python solution_cf.py --input_path train.parquet --output_path submission.csv --model slim
```

To use ALS/iALS:

```bash
python solution_cf.py --input_path train.parquet --output_path submission.csv --model als
```

The default value of `--top_k` is `10`, which matches MAP@10. The output contains one row per recommended user-item pair, ranked from highest to lowest relevance score within each user's group.

### 3. Choose a metrics output path

```bash
python solution_cf.py --input_path train.parquet --test_path validation.parquet --output_path submission.csv --metrics_path metrics.csv
```

## Output files

### Recommendations: `submission.csv`

The CSV has two columns:

```csv
user_id,recs
123,4567
123,8910
123,1122
456,7788
```

Each row contains one recommended `item_id` in the `recs` column. Rows are grouped by user, with each user's recommendations ordered by model score. The script filters training-history items from recommendations where the training history is known.

### Evaluation: `submission_metrics.csv`

The metrics file includes one row per successfully evaluated model and these columns:

- `Model`
- `MAP@K`
- `Precision@K`
- `Recall@K`
- `NDCG@K`
- `CatalogCoverage`
- `EvalUsers`

Here, `K` is controlled by `--top_k` and defaults to 10.

## Metrics

- **MAP@K (Mean Average Precision):** the primary ranking metric. It rewards relevant items appearing near the top of a recommendation list.
- **Precision@K:** the fraction of the first K recommendations that are relevant.
- **Recall@K:** the fraction of the user's relevant hold-out items that appear in the first K recommendations.
- **NDCG@K:** a ranking metric that discounts relevant items appearing lower in the list.
- **CatalogCoverage:** the fraction of items in the training catalog that appear in at least one evaluated user's recommendation list.

For evaluation, distinct user-item pairs in the hold-out file are treated as relevant unless the row is marked as a dislike. This is a practical evaluation assumption, not a definitive judgment about a user's preference. Make sure it matches your assignment's evaluation protocol.

The script also reports the proportion of positive hold-out interactions whose items exist in the training catalog. Items never present in training cannot be recommended by these models.

## Parameter tuning

Start with a baseline run and change one parameter at a time on a **validation** split. Use the final test split only after choosing a model and its settings.

### EASE

```bash
python solution_cf.py --input_path train.parquet --test_path validation.parquet --output_path submission.csv --model ease --ease_max_items 1500 --ease_reg 500
```

- `--ease_max_items`: number of popular items used to fit EASE (default: `1500`).
- `--ease_reg`: diagonal regularization (default: `500`).

EASE constructs a dense item-item matrix. Memory use grows approximately quadratically with the number of selected items, and matrix inversion can be expensive. Increase `--ease_max_items` cautiously.

### SLIM

```bash
python solution_cf.py --input_path train.parquet --test_path validation.parquet --output_path submission.csv --model slim --slim_max_items 500 --slim_alpha 0.001 --slim_l1_ratio 0.05 --slim_neighbors 100
```

- `--slim_max_items`: number of popular items used for training (default: `500`).
- `--slim_alpha`: ElasticNet regularization strength (default: `0.001`).
- `--slim_l1_ratio`: balance of L1 and L2 regularization (default: `0.05`). Larger values generally produce sparser coefficients.
- `--slim_neighbors`: maximum number of retained coefficients per target item (default: `100`).
- `--slim_max_iter`: maximum ElasticNet iterations per target item (default: `1000`).
- `--slim_tol`: ElasticNet convergence tolerance (default: `0.001`).
- `--slim_positive` / `--no-slim_positive`: whether to constrain coefficients to be non-negative (default: enabled).

SLIM trains a separate regression for every selected target item. Training time can become substantial as `--slim_max_items` increases.

### ItemKNN

- `--knn_max_items`: item catalog limit (default: `5000`).
- `--knn_neighbors`: retained neighbors per item (default: `100`).

### ALS / iALS and TruncatedSVD

- `--factors`: latent dimensionality (default: `64`).
- `--iterations`: ALS iterations (default: `20`).
- `--regularization`: ALS regularization (default: `0.05`).
- `--alpha`: confidence scaling for implicit ALS (default: `20.0`).

### General

- `--top_k`: number of recommendations and metric cutoff (default: `10`).
- `--model`: selected output model: `top_popular`, `popularity`, `itemknn`, `ease`, `slim`, `als`, `svd`, or `auto`.
- `--metrics_path`: custom path for the metrics CSV.

Use `python solution_cf.py --help` to see all options.

> **Validation warning:** `--model auto` chooses the model with the highest MAP@K on the supplied hold-out file. Use this only with a validation set for model selection. Selecting a model on the final test set makes that test score an optimistic estimate of generalization.

## Recommended experiment workflow

1. Create a validation split that represents the assignment's intended evaluation protocol.
2. Run all models with default settings and save the metrics table.
3. Select one or two promising models by MAP@10, while checking Recall@10 and CatalogCoverage as secondary indicators.
4. Tune a small number of parameters on validation data. Record every run and avoid changing multiple parameters at once at first.
5. Choose the model and parameter settings before evaluating on a separate final test split.
6. Train the selected model using the allowed training data and generate the submission CSV.

## Limitations and interpretation

- There are no pre-reported benchmark results; results depend on the data split and chosen parameters.
- EASE and SLIM are restricted to the most popular items by default to control computation.
- The popularity baseline counts interaction rows, which may include negative feedback rows, to stay close to the original baseline implementation.
- ALS uses heuristic confidence weights for implicit feedback. Those weights should be validated rather than interpreted as calibrated probabilities.
- Test interactions whose items are absent from the training catalog cannot be recovered by these item-ranking models.

## Data and licensing

The data source is [VK-LSVD on Hugging Face](https://huggingface.co/datasets/deepvk/VK-LSVD). Review the dataset's own license and terms before using or redistributing its contents. This repository does not include the dataset files.

No software license is declared by this repository template. Before publishing the project for reuse, choose and add a license that matches your intentions and any applicable course or institutional requirements.
