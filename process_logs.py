import argparse
import click
from pathlib import Path
from typing import List

from processtransformer.data.processor import LogsDataProcessor
from processtransformer import constants


def _from_xes_to_csv(dataset_path: str):
    import pm4py

    log = pm4py.read_xes(dataset_path)
    pd = pm4py.convert_to_dataframe(log)
    name = Path(dataset_path).name.split(".")[0] + ".csv"
    pd.to_csv(name, index=False)
    return name


def process_dataset(
    dataset_name: str, main_columns: List[str], context_columns: List[str]
):
    path = f"./datasets/{dataset_name}"
    if "xes" in dataset_name:
        path = _from_xes_to_csv(path)
    data_processor = LogsDataProcessor(
        name=Path(path).stem,
        filepath=path,
        columns=main_columns + context_columns,
        dir_path="datasets",
        pool=4,
    )
    data_processor.process_logs(
        task=constants.Task.NEXT_ACTIVITY, sort_temporally=False
    )


@click.command()
@click.argument("dataset_name", type=str)
@click.argument("main_columns", nargs=3)
@click.argument("context_columns", nargs=-1)
def main(dataset_name: str, main_columns: List[str], context_columns: List[str]):
    """
    Process event log datasets for process mining.

    \b
    Parameters:
    - dataset_name: Name of the dataset file located in the 'datasets' folder (either .xes.gz or .csv).
    - main_columns: List of main columns in order: 'Case ID', 'Activity', 'Complete Timestamp'.
    - context_columns: List of context columns (categorical values only).
    """
    process_dataset(dataset_name, list(main_columns), list(context_columns))


if __name__ == "__main__":
    main()
