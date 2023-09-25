import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from pandas import DataFrame

from phd_model_evaluations.cli.draw.draw_lm_gap_features_plot_arguments import LMGapFeaturesDrawPlotArguments
from phd_model_evaluations.data.feature.aggregated_feature import AggregatedFeature
from phd_model_evaluations.data.visualization.visualization_type import VisualizationType

LOGGER = logging.getLogger(__name__)


def draw_lm_gap_features_histogram(
    aggregated_lm_gap_features: AggregatedFeature, draw_arguments: LMGapFeaturesDrawPlotArguments
) -> None:
    LOGGER.info(f"Drawing histogram for: {aggregated_lm_gap_features.name}")
    draw_option = aggregated_lm_gap_features.draw_option
    title_x = "axis_x"

    data_dict: Dict[str, List[int]] = {name: [] for name in [title_x]}
    aggregated_data = [[int(key), value] for key, value in aggregated_lm_gap_features.data.items()]
    aggregated_data.sort(key=lambda x: x[0])
    for value_x, value_y in aggregated_data:
        if value_y < draw_arguments.min_value:
            continue
        data_dict[title_x].extend([int(value_x)] * int(value_y))
    data = DataFrame(data_dict)

    # Draw histogram
    plot: Axes = sns.histplot(data=data, x=title_x, stat="count")
    plot.set_xlabel(draw_option.title_x)
    plot.set_ylabel(draw_option.title_y)

    save_file_path = draw_arguments.save_path / f"{aggregated_lm_gap_features.name}.png"
    LOGGER.info(f"Saving histogram in: {save_file_path}")
    figure = plot.get_figure()
    assert figure is not None, "Cannot get figure"
    figure.text(
        0.6,
        0.8,
        f"Minimalna wartość: {int(aggregated_lm_gap_features.statistics.min_value)}\n"
        f"Maksymalna wartość: {int(aggregated_lm_gap_features.statistics.max_value)}\n"
        f"Średnia wartość: {aggregated_lm_gap_features.statistics.avg_value:.2f}"
        f"±{aggregated_lm_gap_features.statistics.std_value:.2f}",
        fontsize=15,
    )
    figure.savefig(save_file_path, bbox_inches="tight")
    plt.clf()


def draw_lm_gap_features_pie(
    aggregated_lm_gap_features: AggregatedFeature, draw_arguments: LMGapFeaturesDrawPlotArguments
) -> None:
    LOGGER.info(f"Drawing pie chart for: {aggregated_lm_gap_features.name}")
    draw_option = aggregated_lm_gap_features.draw_option

    data_and_labels = []
    total = aggregated_lm_gap_features.total
    for value_x, value_y in sorted(aggregated_lm_gap_features.data.items(), key=lambda x: x[0]):
        value_x = str(value_x)
        data_and_labels.append((draw_option.map_key_to_label_name.get(value_x, value_x), value_y))
        total -= value_y

    if total > 0:
        data_and_labels.append((draw_option.other_label_name, total))

    # Sort data and labels with descending order (greater value will be first)
    data_and_labels.sort(key=lambda x: x[1], reverse=True)

    # Add percentage to the labels
    labels_with_percentage = []
    total_values = float(sum(data for _, data in data_and_labels))
    for label, value in data_and_labels:
        percentage = float(value) / total_values * 100.0
        if percentage < 0.01:
            labels_with_percentage.append(f"{label} (<0.01%)")
        else:
            labels_with_percentage.append(f"{label} ({percentage:.2f}%)")

    fig, ax = plt.subplots(figsize=(draw_arguments.figure_height, draw_arguments.figure_width))
    wedges, _ = ax.pie([data for _, data in data_and_labels])

    ax.legend(wedges, labels_with_percentage, loc="center left", bbox_to_anchor=(0.0, 0.9), fontsize=15)

    save_file_path = draw_arguments.save_path / f"{aggregated_lm_gap_features.name}.png"
    LOGGER.info(f"Saving pie chart in: {save_file_path}")
    plt.savefig(save_file_path, bbox_inches="tight")
    plt.clf()


def draw_lm_gap_features_plot(
    aggregated_lm_gap_features: AggregatedFeature, draw_arguments: LMGapFeaturesDrawPlotArguments
) -> None:
    sns.set_theme(
        context="poster",
        palette="bright",
        rc={"figure.figsize": (draw_arguments.figure_height, draw_arguments.figure_width)},
    )

    draw_option = aggregated_lm_gap_features.draw_option
    if draw_option.visualization_type == VisualizationType.histogram:
        draw_lm_gap_features_histogram(aggregated_lm_gap_features, draw_arguments)

    elif draw_option.visualization_type == VisualizationType.pie:
        draw_lm_gap_features_pie(aggregated_lm_gap_features, draw_arguments)

    else:
        enum_str = str(draw_option.visualization_type)
        raise RuntimeError(f"Unsupported plot type: {enum_str}")
