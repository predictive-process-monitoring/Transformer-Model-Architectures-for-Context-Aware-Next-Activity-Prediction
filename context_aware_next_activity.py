import json
import time
import click
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.model_selection import KFold
from uuid import uuid4

from processtransformer import constants
from processtransformer.models import transformer
from processtransformer.data.loader import LogsDataLoader

import warnings

warnings.filterwarnings("ignore")


def evaluate_next_activity_with_context(
    dataset_name: str,
    context_columns: List[str],
    model_type: str,
    prefix_length: str,
    n_splits: int = 5,
    epochs: int = 10,
) -> None:
    data_loader = LogsDataLoader(name=dataset_name)
    (
        all_data_df,
        x_word_dict,
        context_dict,
        y_word_dict,
        max_case_length,
        vocab_size,
        context_vocab_size,
        num_output,
        input_dimension,
    ) = data_loader.load_data(
        constants.Task.NEXT_ACTIVITY, context_columns, prefix_length
    )

    print(f"K prefix length is {input_dimension}")
    print(f"Max case length is {max_case_length}")

    kf = KFold(n_splits=n_splits, shuffle=False)

    learning_rate = 0.001
    batch_size = 256

    # Define lists to store evaluation metrics across all folds
    all_accuracies, all_fscores, all_precisions, all_recalls, all_len_test_indexes = (
        [],
        [],
        [],
        [],
        [],
    )

    training_times = []
    for train_index, test_index in kf.split(all_data_df):
        train_df = all_data_df.iloc[train_index]
        test_df = all_data_df.iloc[test_index]

        # Prepare training examples for next activity prediction task
        (
            train_token_x,
            train_token_context,
            train_token_y,
        ) = data_loader.prepare_data_next_activity(
            train_df,
            x_word_dict,
            context_dict,
            y_word_dict,
            input_dimension,
        )

        # Create and compile a transformer model
        transformer_model = transformer.get_next_activity_model(
            max_case_length=input_dimension,
            context_size=len(train_token_context),
            vocab_size=vocab_size,
            output_dim=num_output,
            context_vocab_size=context_vocab_size,
            model_type=model_type,
        )

        transformer_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        inputs = [train_token_x] + train_token_context
        t0 = time.time()
        # Train the model
        transformer_model.fit(
            inputs,
            train_token_y,
            epochs=epochs,
            batch_size=batch_size,
        )
        training_time = time.time() - t0
        training_times.append(training_time)

        # Evaluate over all the prefixes (k) and save the results
        k, accuracies, fscores, precisions, recalls, len_test_indexes = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for i in range(max_case_length):
            test_data_subset = test_df[test_df["k"] == i]
            if len(test_data_subset) > 0:
                (
                    test_token_x,
                    test_token_context,
                    test_token_y,
                ) = data_loader.prepare_data_next_activity(
                    test_data_subset,
                    x_word_dict,
                    context_dict,
                    y_word_dict,
                    input_dimension,
                )

                inputs = [test_token_x] + test_token_context

                y_pred = np.argmax(
                    transformer_model.predict(inputs),
                    axis=1,
                )
                accuracy = metrics.accuracy_score(test_token_y, y_pred)
                precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                    test_token_y, y_pred, average="weighted"
                )

                cm = metrics.confusion_matrix(test_token_y, y_pred)
                disp = metrics.ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                )
                disp.plot()
                plt.show()

                k.append(i)
                len_test_indexes.append(len(test_data_subset))
                accuracies.append(accuracy)
                fscores.append(fscore)
                precisions.append(precision)
                recalls.append(recall)
            else:
                k.append(i)
                len_test_indexes.append(0)
                accuracies.append(0)
                fscores.append(0)
                precisions.append(0)
                recalls.append(0)
        k.append(i + 1)
        accuracies.append(np.average(accuracies, weights=len_test_indexes))
        fscores.append(np.average(fscores, weights=len_test_indexes))
        precisions.append(np.average(precisions, weights=len_test_indexes))
        recalls.append(np.average(recalls, weights=len_test_indexes))
        len_test_indexes.append(np.sum(len_test_indexes))

        # Append metrics for this fold to the overall lists
        all_accuracies.append(accuracies)
        all_fscores.append(fscores)
        all_precisions.append(precisions)
        all_recalls.append(recalls)
        all_len_test_indexes.append(len_test_indexes)
        print(f"Overall Accuracy {accuracies[-1]}")

    # Compute average performance metrics across all folds
    index_scores = {}
    for i in range(max_case_length + 1):
        weights = [len_test_indexes[i] for len_test_indexes in all_len_test_indexes]
        accuracy = np.average(
            [accuracy[i] for accuracy in all_accuracies], weights=weights
        )
        precision = np.average(
            [precision[i] for precision in all_precisions], weights=weights
        )
        fscore = np.average([fscore[i] for fscore in all_fscores], weights=weights)
        recall = np.average([recall[i] for recall in all_recalls], weights=weights)
        index_scores[str(i + 1)] = {
            "size": int(np.sum(weights)),
            "accuracy": float(accuracy),
            "fscore": float(fscore),
            "precision": float(precision),
            "recall": float(recall),
        }

    results = {}
    results.update(
        {
            str(uuid4()): {
                "dataset_name": dataset_name,
                "context columns": context_columns,
                "training_time": np.mean(training_times),
                "index_scores": index_scores,
                "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "n_splits": n_splits,
                "model_type": model_type,
                "prefix_length": input_dimension,
            }
        }
    )
    pprint(results)
    current_results = {}
    if Path("results.json").is_file():
        with open("results.json", "r") as f:
            current_results = json.load(f)
    current_results.update(results)
    with open("results.json", "w+") as f:
        json.dump(
            current_results,
            f,
            indent=4,
            sort_keys=True,
        )

model_types = [
    "transformer-single",
    "transformer-multiple",
    "dense-single",
    "dense-multiple",
    "bi-lstm-single",
    "bi-lstm-multiple",
]

prefix_lengths = ["Min", "One-Quarter", "Middle", "Max"]


@click.command()
@click.argument("dataset_name", type=str)
@click.argument("context_columns", nargs=-1)
@click.option(
    "--model-type",
    type=click.Choice(model_types),
    required=True,
    help="Type of the model to be used for evaluation.",
)
@click.option(
    "--prefix-length",
    type=click.Choice(prefix_lengths),
    required=True,
    help="Prefix length for evaluation.",
)
@click.option(
    "--n-splits",
    type=int,
    default=5,
    show_default=True,
    help="Number of splits for cross-validation.",
)
@click.option(
    "--epochs",
    type=int,
    default=50,
    show_default=True,
    help="Number of epochs for training the model.",
)
def main(
    dataset_name: str,
    context_columns: List[str],
    model_type: str,
    prefix_length: str,
    n_splits: int,
    epochs: int,
):
    """
    Evaluate next activity prediction with context.

    \b
    Parameters:
    - dataset_name: Name of the dataset folder located in the 'datasets' directory.
    - context_columns: List of context columns (categorical values only).
    """
    evaluate_next_activity_with_context(
        dataset_name, list(context_columns), model_type, prefix_length, n_splits, epochs
    )


if __name__ == "__main__":
    main()
