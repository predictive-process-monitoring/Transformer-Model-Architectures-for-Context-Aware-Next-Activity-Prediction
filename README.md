# Transformer Model Architectures for Context-Aware Next Activity Prediction

The aim of this project is to evaluate the impact of different model architectures on next activity prediction in process mining. This includes various model types such as transformers, LSTMs, and dense networks, both with and without hierarchical structures. The hierarchical models are indicated by "multiple" in the model type. Additionally, the project explores the influence of different prefix lengths on the prediction performance.

## Overview

The process involves two main steps:
1. **Processing your dataset** using the first CLI command.
2. **Evaluating the dataset** with context columns using the second CLI command.

## Step 1: Processing Your Dataset

Before evaluating, you need to process your dataset. The CLI command for this is defined in `process_logs.py`.

### Usage

```bash
python process_logs.py dataset_name "main_columns" "context_columns"
```

### Parameters

- `dataset_name` (str): Name of the dataset file located in the `datasets` folder (either `.xes.gz` or `.csv`).
- `main_columns` (List[str]): List of main columns in the following order:
  1. `Case ID`
  2. `Activity`
  3. `Timestamp`
- `context_columns` (List[str]): List of context columns (categorical values only).

### Example

Suppose you have a dataset named `event_log.xes.gz` in the `datasets` folder. The main columns are `Case ID`, `Activity`, and `Complete Timestamp`, and the context columns are `Resource`, `customer`. You can process the dataset by running:

```bash
python process_logs.py event_log.xes.gz "Case ID" "Activity" "Complete Timestamp" "Resource" "customer"
```

This command will convert the `.xes.gz` file to `.csv`, process the logs, and prepare them for prediction algorithms.

## Step 2: Evaluating Your Dataset

Once the dataset is processed, you can evaluate it using different context columns with the second CLI command defined in `evaluate.py`.

### Usage

```bash
python context_aware_next_activity.py dataset_name context_columns --model-type MODEL_TYPE --prefix-length PREFIX_LENGTH [--n_splits N_SPLITS] [--epochs EPOCHS]
```

### Parameters

- `dataset_name` (str): Name of the dataset folder located in the `datasets` directory.
- `context_columns` (List[str]): List of context columns (categorical values only).
- `--model-type` (str): Type of the model to be used for evaluation. Possible values:
  - `transformer-single`
  - `dense-single`
  - `dense-multiple`
  - `transformer-multiple`
  - `bi-lstm-single` 
  - `bi-lstm-multiple`
- `--prefix-length` (str): Prefix length for evaluation. Possible values:
  - `Min`
  - `One-Quarter`
  - `Middle`
  - `Max`
- `--n_splits` (int, optional): Number of splits for cross-validation (default is 5).
- `--epochs` (int, optional): Number of epochs for training the model (default is 10).

### Example

To evaluate the dataset named `helpdesk` in the `datasets` folder with context column `Resource`, using a `transformer-single` model and `Middle` prefix length, run:

```bash
python context_aware_next_activity.py helpdesk "Resource" --model-type transformer-single --prefix-length Middle
```

## Experiment Script

An experiment to evaluate the effect of different architecture types and prefix lengths is provided in `experiment.py`. The script iterates over all combinations of model types and prefix lengths, evaluating each configuration with and without context columns.

### Running the Experiment

To run the experiment, simply execute the `experiment.py` script:

```bash
python experiment.py
```

This will evaluate the effect of different model architectures and prefix lengths on the next activity prediction, iterating over all specified configurations and context columns.

## Notes

By following these steps, you can process and evaluate your datasets effectively, gaining insights into the performance of different model architectures and prefix lengths in the context-aware next activity prediction.
