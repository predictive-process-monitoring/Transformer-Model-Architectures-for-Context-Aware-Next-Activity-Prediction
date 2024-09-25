import io
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import utils
from sklearn import preprocessing

from ..constants import Task


def _is_float(number: str):
    try:
        if number in ["nan", "nil"]:
            return False
        float(number)
        return True
    except ValueError:
        return False


class LogsDataLoader:
    def __init__(self, name, dir_path="./datasets"):
        """Provides support for reading and
            pre-processing examples from processed logs.
        Args:
            name: str: name of the dataset as used during processing raw logs
            dir_path: str: Path to dataset directory
        """
        self._dir_path = f"{dir_path}/{name}/processed"

    def prepare_data_next_activity(
        self,
        df,
        x_word_dict,
        context_word_dict,
        y_word_dict,
        max_case_length,
        context_history=True,
    ):

        x = df["prefix"].values
        context_df = df[df.columns.values[4:]]
        y = df["next_act"].values
        context_values = [context_df[column].values for column in context_df]

        token_x = list()
        for _x in x:
            token_x.append(
                list(
                    reversed(
                        list(reversed([x_word_dict[s] for s in _x.split()]))[
                            :max_case_length
                        ]
                    )
                )
            )

        token_context = []
        for column in context_df:
            token_c = list()
            c_word_dict = context_word_dict[f"{column}_word_dict"]
            if column not in ["org:resource", "dismissal", "amount"] and any(
                [_is_float(key) for key in list(c_word_dict.keys())]
            ):
                time_scaler = preprocessing.MinMaxScaler()
                if context_history:
                    for _c in context_values[len(token_context)]:
                        pass
                    token_context.append(
                        time_scaler.fit_transform(
                            np.nan_to_num(context_values[len(token_context)])
                        ).astype(np.float32)
                    )
                else:
                    token_context.append(
                        time_scaler.fit_transform(
                            np.nan_to_num(context_values[len(token_context)]).reshape(
                                -1, 1
                            )
                        ).astype(np.float32)
                    )
            else:
                if context_history:
                    for _c in context_values[len(token_context)]:
                        token_c.append(
                            list(
                                reversed(
                                    list(
                                        reversed([c_word_dict[c] for c in _c.split()])
                                    )[:max_case_length]
                                )
                            )
                        )
                    token_c = tf.keras.preprocessing.sequence.pad_sequences(
                        token_c, maxlen=max_case_length
                    )
                    token_context.append(np.array(token_c, dtype=np.float32))
                else:
                    for _c in context_values[len(token_context)]:
                        token_c.append(c_word_dict[str(_c).split()[-1]])
                    token_context.append(np.array(token_c, dtype=np.float32))

        token_y = list()
        for _y in y:
            token_y.append(y_word_dict[_y])
        # token_y = np.array(token_y, dtype = np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length
        )

        token_x = np.array(token_x, dtype=np.float32)
        token_y = np.array(token_y, dtype=np.float32)

        return token_x, token_context, token_y

    def prepare_data_next_time(
        self,
        df,
        x_word_dict,
        k_prefix_length,
        time_scaler=None,
        y_scaler=None,
        shuffle=True,
    ):

        x = df["prefix"].values
        time_x = df[["recent_time", "latest_time", "time_passed"]].values.astype(
            np.float32
        )
        y = df["next_time"].values.astype(np.float32)
        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(time_x).astype(np.float32)

        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=k_prefix_length
        )

        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return token_x, time_x, y, time_scaler, y_scaler

    def prepare_data_remaining_time(
        self,
        df,
        x_word_dict,
        max_case_length,
        time_scaler=None,
        y_scaler=None,
        shuffle=True,
    ):

        x = df["prefix"].values
        time_x = df[["recent_time", "latest_time", "time_passed"]].values.astype(
            np.float32
        )
        y = df["remaining_time_days"].values.astype(np.float32)

        if shuffle:
            x, time_x, y = utils.shuffle(x, time_x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        if time_scaler is None:
            time_scaler = preprocessing.StandardScaler()
            time_x = time_scaler.fit_transform(time_x).astype(np.float32)
        else:
            time_x = time_scaler.transform(time_x).astype(np.float32)

        if y_scaler is None:
            y_scaler = preprocessing.StandardScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
        else:
            y = y_scaler.transform(y.reshape(-1, 1)).astype(np.float32)

        token_x = tf.keras.preprocessing.sequence.pad_sequences(
            token_x, maxlen=max_case_length
        )

        token_x = np.array(token_x, dtype=np.float32)
        time_x = np.array(time_x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        return token_x, time_x, y, time_scaler, y_scaler

    def get_max_case_length(self, train_x):
        train_token_x = list()
        for _x in train_x:
            train_token_x.append(len(_x.split()))
        return max(train_token_x)

    def load_data(self, task, context_columns=None, prefix_length: str = "Max"):
        if context_columns is None:
            context_columns = []
        if task not in (Task.NEXT_ACTIVITY, Task.NEXT_TIME, Task.REMAINING_TIME):
            raise ValueError("Invalid task.")

        train_df = pd.read_csv(f"{self._dir_path}/{task.value}_train.csv")
        test_df = pd.read_csv(f"{self._dir_path}/{task.value}_test.csv")

        main_columns = list(train_df.columns.values[:4]) + context_columns
        train_df = train_df[main_columns]
        test_df = test_df[main_columns]

        with open(f"{self._dir_path}/metadata.json", "r") as json_file:
            metadata = json.load(json_file)

        x_word_dict = metadata["x_word_dict"]
        y_word_dict = metadata["y_word_dict"]
        del metadata["x_word_dict"]
        del metadata["y_word_dict"]
        context_dict = metadata
        vocab_size = len(x_word_dict)
        context_vocab_size = 0
        for context in context_columns:
            context_vocab_size = context_vocab_size + len(
                context_dict[f"{context}_word_dict"]
            )
        total_classes = len(y_word_dict)
        all_data = pd.concat([train_df, test_df])
        max_case_length = self.get_max_case_length(train_df["prefix"].values)

        input_middle = int(all_data.groupby("case_id")["k"].max().mean()) + 2

        if prefix_length == "Min":
            input_dimension = 1
        elif prefix_length == "One-Quarter":
            input_dimension = int(input_middle / 2)
            if input_dimension == 1:
                input_dimension = 2
        elif prefix_length == "Middle":
            input_dimension = input_middle
        elif prefix_length == "Max":
            input_dimension = max_case_length
        return (
            all_data,
            x_word_dict,
            context_dict,
            y_word_dict,
            max_case_length,
            vocab_size,
            context_vocab_size,
            total_classes,
            input_dimension,
        )
