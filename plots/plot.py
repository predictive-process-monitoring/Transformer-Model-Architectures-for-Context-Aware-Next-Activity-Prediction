from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


sns.set_context("talk", font_scale=1.2)
sns.set_style("whitegrid")


def group_plot(
    df,
    group_col: str,
    value_col: str,
    hue: str = None,
    plot_type: str = "bar",
    title: str = None,
    order: List[str] = None,
    col: str = None,
    col_order: List[str] = None,
    y_label: str = "Accuracy",
    horizontal_value: float = None,
    x_label: str = None,
):
    high_contrast_palettes = {
        "Set1": sns.color_palette("Set1"),
        "Set2": sns.color_palette("Set2"),
        "Set3": sns.color_palette("Set3"),
        "Dark2": sns.color_palette("Dark2"),
        "Paired": sns.color_palette("Paired"),
    }

    # Choose a palette
    selected_palette = high_contrast_palettes["Set1"]

    g = sns.catplot(
        data=df,
        x=group_col,
        y=value_col,
        hue=hue,
        col=col,
        kind=plot_type,
        palette=selected_palette,
        alpha=0.8,
        order=order,
        hue_order=col_order,
        aspect=1.2,  # Adjust aspect ratio
        height=12,  # Adjust height of the plots
    )

    # Add vertical line to each subplot in the FacetGrid
    if horizontal_value:
        for ax in g.axes.flatten():
            ax.axhline(
                y=horizontal_value,
                color="red",
                linestyle="--",
                linewidth=2,
                label="baseline",
            )

    # Set labels and title
    g.set_axis_labels(x_label, y_label)
    g.set_titles(title)
    # plt.tight_layout()

    # Customize fonts and style for publication
    plt.xlabel(x_label, fontsize=16, weight="bold")
    plt.ylabel(y_label, fontsize=16, weight="bold")
    plt.suptitle(title, fontsize=18, weight="bold")
    for ax in g.axes.flatten():
        ax.tick_params(labelsize=14)
    # plt.show()
    plt.savefig(f'./{title.replace(" ", "_").lower()}.png', bbox_inches="tight")
    grouping = [group_col]
    if hue:
        grouping.append(hue)
    if col:
        grouping.append(col)
    df.groupby(grouping)[value_col].describe().to_csv(
        f'./{title.replace(" ", "_").lower()}.csv'
    )
    plt.close()


def correlation(df: pd.DataFrame, x, y):
    sns.scatterplot(
        df,
        x=x,
        y=y,
        alpha=0.8,
    )
    plt.title("ATI Score mean with Questions")
    plt.show()
    plt.savefig(f"analysis_data/scatter_{x}_{y}.png", bbox_inches="tight")
    plt.close()


def plot(
    dataset_name: str,
    prefix_length: int,
    prefix_index: int,
    title: str,
    base_accuracy: float,
    base_training: float,
):
    df_prefix = pd.read_csv("../analysis/all_data_index_40.csv")
    model_type_mapping = {
        "transformer-single": "Single TE",
        "transformer-multiple": "Hierarchical TE",
        "bi-lstm-single": "Single biLSTM",
        "bi-lstm-multiple": "Hierarchical biLSTM",
        "dense-single": "Single ReLU",
        "dense-multiple": "Hierarchical ReLU",
    }
    value_mapping = {
        "Without prefix": "0",
        "Average Case Length / 2": "L\u0305" + "/ 2",
        "Average Case Length": "L\u0305",
        "Max Case Length": "Lmax",
    }
    df_prefix["model_type"] = df_prefix["model_type"].replace(model_type_mapping)
    df_prefix.rename(columns={"model_type": "Architecture Type"}, inplace=True)
    df_prefix["prefix_length"] = df_prefix["prefix_length"].replace(value_mapping)
    df_prefix = df_prefix[
        (df_prefix["dateset"] == dataset_name)
        & (df_prefix["context-size"] == 1)
        & (df_prefix["prefix_index"] <= prefix_index)
        & (df_prefix["prefix_length"] == prefix_length)
    ]
    group_plot(
        df_prefix,
        value_col="accuracy",
        group_col="prefix_index",
        hue="Architecture Type",
        title=f"Prefix Length Prediction in {title}",
        plot_type="point",
        horizontal_value=base_accuracy,
        col_order=[
            "Single TE",
            "Hierarchical TE",
            "Single biLSTM",
            "Hierarchical biLSTM",
            "Single ReLU",
            "Hierarchical ReLU",
        ],
        x_label="k-Prefix Length",
    )

    df_all = pd.read_csv("../analysis/all_data.csv")
    df_all["model_type"] = df_all["model_type"].replace(model_type_mapping)
    df_all.rename(columns={"model_type": "Architecture Type"}, inplace=True)
    df_all["prefix_length"] = df_all["prefix_length"].replace(value_mapping)
    df_all_without_first_index = df_all[
        (df_all["dateset"] == dataset_name) & (df_all["context-size"] == 1)
    ]

    df_all_with_first_index = df_all[
        (df_all["dateset"] == dataset_name)
        & (df_all["context-size"] == 1)
        & (df_all["prefix_length"] != 0)
    ]

    group_plot(
        df_all_without_first_index,
        value_col="accuracy",
        group_col="prefix_length",
        hue="Architecture Type",
        title=f"Accuracy in {title} Without First Index",
        plot_type="point",
        horizontal_value=base_accuracy,
        order=[
            "L\u0305" + "/ 2",
            "L\u0305",
            "Lmax",
        ],
        col_order=[
            "Single TE",
            "Hierarchical TE",
            "Single biLSTM",
            "Hierarchical biLSTM",
            "Single ReLU",
            "Hierarchical ReLU",
        ],
        x_label="k-Prefix Length",
    )

    group_plot(
        df_all_with_first_index,
        value_col="accuracy",
        group_col="prefix_length",
        hue="Architecture Type",
        title=f"Accuracy in {title}",
        plot_type="point",
        horizontal_value=base_accuracy,
        order=[
            "0",
            "L\u0305" + "/ 2",
            "L\u0305",
            "Lmax",
        ],
        col_order=[
            "Single TE",
            "Hierarchical TE",
            "Single biLSTM",
            "Hierarchical biLSTM",
            "Single ReLU",
            "Hierarchical ReLU",
        ],
        x_label="k-Prefix Length",
    )

    group_plot(
        df_all_with_first_index,
        value_col="training_time",
        group_col="prefix_length",
        hue="Architecture Type",
        title=f"Training Time in {title}",
        plot_type="point",
        horizontal_value=base_training,
        y_label="Training Time in Seconds",
        order=[
            "0",
            "L\u0305" + "/ 2",
            "L\u0305",
            "Lmax",
        ],
        col_order=[
            "Single TE",
            "Hierarchical TE",
            "Single biLSTM",
            "Hierarchical biLSTM",
            "Single ReLU",
            "Hierarchical ReLU",
        ],
        x_label="k-Prefix Length",
    )


def plot_all():
    plot(
        "bpi_challenge_2020",
        8,
        15,
        "BPI Challenge 2020",
        0.8201808721506442,
        181.57269797325134,
    )

    plot(
        "bpi_challenge_2017",
        19,
        38,
        "BPI Challenge 2017",
        0.8749483239889438,
        4489.325288534164,
    )

    plot("helpdesk", 4, 7, "Helpdesk", 0.7947632112608851, 51.020753955841066)
    plot("sepsis", 7, 20, "Sepsis", 0.6179115044247787, 44.55631065368652)
    plot(
        "bpi_challenge_2012",
        20,
        30,
        "BPI Challenge 2012",
        0.8462566737585806,
        945.0106904029847,
    )
    plot(
        "road_traffic_fine_management_process",
        3,
        6,
        "Road Traffic Fine Management Process",
        0.8118243736317198,
        1217.6870047569275,
    )


if __name__ == "__main__":
    # plot_all()
    dataset_mapping = {
        "helpdesk": "Helpdesk",
        "road_traffic_fine_management_process": "TrafficFines",
        "sepsis": "Sepsis",
        "bpi_challenge_2020": "BPIC20p",
        "bpi_challenge_2012": "BPIC12",
        "bpi_challenge_2017": "BPIC17",
    }
    df_prefix = pd.read_csv("../analysis/all_data_embedding.csv")
    df_prefix["Event Log"] = df_prefix["dateset"].replace(dataset_mapping)
    group_plot(
        df_prefix,
        "Embedding",
        "accuracy",
        "Event Log",
        "point",
        "Comparison of Attribute Embedding's Impact in Accuracy",
        x_label="Embedding",
        col_order=['Helpdesk', 'BPIC12', 'BPIC17', 'BPIC20p', 'Sepsis', 'TrafficFines']
    )
