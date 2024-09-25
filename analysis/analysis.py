import json

import pandas as pd

prefix_length = {
    "helpdesk": 4,
    "bpi_challenge_2012": 20,
    "bpi_challenge_2017": 38,
    "bpi_challenge_2020": 8,
    "sepsis": 14,
    "road_traffic_fine_management_process": 3,
}


def latest_index(indexes: dict):
    last_index = 0
    for index in indexes.keys():
        if int(index) > last_index:
            last_index = int(index)
    return str(last_index)


def to_df():
    with open("../results.json") as f:
        data = json.load(f)

    results = []
    for result in data.values():
        if len(result["context columns"]) == 1:
            token_embedding = (
                "With Embedding"
                if "without" not in result["model_type"]
                else "Without Embedding"
            )
            avg_scores = result["index_scores"]
            latest = latest_index(avg_scores)
            general_avg_scores = avg_scores[latest]
            average_case_length = prefix_length[result["dataset_name"]]
            prefix = result["prefix_length"]
            if not prefix == average_case_length:
                continue
            if prefix == average_case_length:
                prefix = "Average Case Length"
            elif prefix > average_case_length:
                prefix = "Max Case Length"
            elif prefix == 1:
                prefix = "Without prefix"
            elif prefix < average_case_length:
                prefix = "Average Case Length / 2"
            result_data = {
                "context-size": len(result["context columns"]),
                "context_columns": result["context columns"],
                "dateset": result["dataset_name"],
                "accuracy": general_avg_scores["accuracy"],
                "precision": general_avg_scores["precision"],
                "fscore": general_avg_scores["fscore"],
                "recall": general_avg_scores["recall"],
                "training_time": result["training_time"],
                "model_type": result["model_type"],
                "prefix_length": prefix,
                "Embedding" : token_embedding
            }
            results.append(result_data)

    df = pd.DataFrame(results)
    df.to_csv("./all_data_embedding.csv")


def to_df_dimension():
    with open("../results.json") as f:
        data = json.load(f)

    results = []
    for result in data.values():
        if result["dataset_name"] in ["helpdesk", "sepsis"]:
            avg_scores = result["index_scores"]
            latest = latest_index(avg_scores)
            general_avg_scores = avg_scores[latest]
            result_data = {
                "context-size": len(result["context columns"]),
                "context_columns": result["context columns"],
                "dateset": result["dataset_name"],
                "accuracy": general_avg_scores["accuracy"],
                "precision": general_avg_scores["precision"],
                "fscore": general_avg_scores["fscore"],
                "recall": general_avg_scores["recall"],
                "training_time": result["training_time"],
                "model_type": result["model_type"],
                "token_type": result["token_type"],
                "input_dimension": result.get("input_dimension", "MAX"),
            }
            results.append(result_data)

    df = pd.DataFrame(results)
    df.to_csv("./all_data_input_dimension.csv")


def to_df_window():
    with open("../results.json") as f:
        data = json.load(f)

    results = []
    for result in data.values():
        if "without" not in result["model_type"]:
            avg_scores = result["index_scores"]
            for prefix, index_score in avg_scores.items():
                if int(prefix) <= 40:
                    result_data = {
                        "context-size": len(result["context columns"]),
                        "context_columns": result["context columns"],
                        "prefix_index": prefix,
                        "dateset": result["dataset_name"],
                        "accuracy": index_score["accuracy"],
                        "precision": index_score["precision"],
                        "fscore": index_score["fscore"],
                        "recall": index_score["recall"],
                        "model_type": result["model_type"],
                        "prefix_length": result["prefix_length"],
                    }
                    results.append(result_data)

    df = pd.DataFrame(results)
    df.to_csv("./all_data_index_40.csv")


if __name__ == "__main__":
    # to_df_window()
    to_df()
    to_df_window()
    # to_df_dimension()
