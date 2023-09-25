import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy
import numpy as np
import seaborn as sns
from matplotlib.figure import FigureBase
from pandas import DataFrame

from phd_model_evaluations.cli.draw.draw_plot_arguments import DrawPlotArguments
from phd_model_evaluations.data.results.aggregated_results import AggregatedResults
from phd_model_evaluations.statistics.generate_probability_statistics import generate_accepted_score_statistics

LOGGER = logging.getLogger(__name__)


def remove_outside_points(
    data: Dict[str, List[Any]], key_to_check: str, keep_below_min: bool = False, keep_above_max: bool = False
) -> Dict[str, List[Any]]:
    total_elements = len(data[key_to_check])
    statistics = generate_accepted_score_statistics(numpy.array(data[key_to_check], dtype=np.float64))

    indexes_to_remove = []
    data_keys = list(data.keys())
    for index in range(total_elements):
        value = data[key_to_check][index]
        if (not keep_below_min and value < statistics.accepted_min_value) or (
            not keep_above_max and value > statistics.accepted_max_value
        ):
            data_to_debug = {key: data[key][index] for key in data_keys}
            LOGGER.debug(
                f"Skipping outside value: {value} where accepted min: {statistics.accepted_min_value}"
                f" and accepted max: {statistics.accepted_max_value} for: {data_to_debug}"
            )
            indexes_to_remove.append(index)

    # Remove indexes
    for index in reversed(indexes_to_remove):
        for key in data_keys:
            del data[key][index]

    LOGGER.info(f"Removed {len(indexes_to_remove)} outside points for key: {key_to_check}")

    return data


def filter_skipped_labels(
    x_labels: List[str], y_labels: List[str], x_data_labels_set: Set[str], y_data_labels_set: Set[str]
) -> Tuple[List[str], List[str]]:
    new_x_labels, skip_x_labels = [], []
    for x_label in x_labels:
        if x_label in x_data_labels_set:
            new_x_labels.append(x_label)
        else:
            skip_x_labels.append(x_label)
    if skip_x_labels:
        LOGGER.info(f"Removed {skip_x_labels} labels from X axis")

    new_y_labels, skip_y_labels = [], []
    for y_label in y_labels:
        if y_label in y_data_labels_set:
            new_y_labels.append(y_label)
        else:
            skip_y_labels.append(y_label)
    if skip_y_labels:
        LOGGER.info(f"Removed {skip_y_labels} labels from Y axis")

    return new_x_labels, new_y_labels


def filter_values(
    data: Dict[str, List[Any]], key_to_check: str, min_value: Optional[float], max_value: Optional[float]
) -> Dict[str, List[Any]]:
    total_elements = len(data[key_to_check])

    indexes_to_remove = []
    data_keys = list(data.keys())
    for index in range(total_elements):
        value = data[key_to_check][index]
        if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
            data_to_debug = {key: data[key][index] for key in data_keys}
            LOGGER.debug(f"Skipping value: {value} where min: {min_value} and max: {max_value} for: {data_to_debug}")
            indexes_to_remove.append(index)

    # Remove indexes
    for index in reversed(indexes_to_remove):
        for key in data_keys:
            del data[key][index]

    LOGGER.info(f"Removed {len(indexes_to_remove)} points below/above threshold for key: {key_to_check}")

    return data


def draw_label_on_top(
    plot: FigureBase,
    model_name_with_x_position: List[Tuple[str, float]],
    y_position: float = 100.0,
    font_size: float = 7.5,
    step_value_for_next_label: float = 0.1,
    max_label_length: int = 16,
) -> None:
    previous_x_positions = 0.0 - step_value_for_next_label
    for model_name, x_positions in model_name_with_x_position:
        # Skip if label overlaps or label is too long
        if x_positions - previous_x_positions < step_value_for_next_label or len(model_name) > max_label_length:
            continue

        previous_x_positions = x_positions
        plot.text(x_positions, y_position, model_name, fontsize=font_size, fontweight="bold", rotation=65.0)


def get_sorted_model_names_with_x_position(
    data: Dict[str, Optional[float]], model_names: List[str]
) -> List[Tuple[str, float]]:
    model_name_with_x_position = []
    for model_name in model_names:
        score = data.get(model_name)
        if score is not None:
            model_name_with_x_position.append((model_name, score))

    model_name_with_x_position.sort(key=lambda x: x[1])
    return model_name_with_x_position


def draw_plot(aggregated_results: AggregatedResults, draw_arguments: DrawPlotArguments) -> None:
    sns.set_theme(
        context="poster",
        palette="bright",
        rc={"figure.figsize": (draw_arguments.figure_height, draw_arguments.figure_width)},
    )

    title_x = draw_arguments.x_name if draw_arguments.x_title is None else draw_arguments.x_title
    title_y = ", ".join(draw_arguments.y_names) if draw_arguments.y_title is None else draw_arguments.y_title
    key_dataset_name = "Dataset name"
    key_model_name = "Model name"
    model_names = draw_arguments.key_names if draw_arguments.key_names else aggregated_results.model_list
    dataset_names = draw_arguments.y_names

    data_x = aggregated_results.results[draw_arguments.x_name]

    data_dict: Dict[str, List[Any]] = {name: [] for name in [title_x, title_y, key_dataset_name, key_model_name]}
    for dataset_name in draw_arguments.y_names:
        dataset_data = aggregated_results.results[dataset_name]
        for model_name in model_names:
            if model_name not in dataset_data or dataset_data[model_name] is None:
                LOGGER.debug(
                    f"Skipping missing score/point for dataset name: {dataset_name} and model name: {model_name}"
                )
                continue
            elif model_name not in data_x:
                LOGGER.debug(
                    f"Skipping missing score/point in X ({draw_arguments.x_name}) axis for model name: {model_name}"
                )
                continue

            data_dict[title_x].append(data_x[model_name])
            data_dict[title_y].append(dataset_data[model_name])
            data_dict[key_dataset_name].append(dataset_name)
            data_dict[key_model_name].append(model_name)
    # Remove outside points
    if draw_arguments.skip_outside_points:
        data_dict = remove_outside_points(
            data_dict,
            title_x,
            keep_below_min=draw_arguments.keep_min_outside_x_points,
            keep_above_max=draw_arguments.keep_max_outside_x_points,
        )
        data_dict = remove_outside_points(
            data_dict,
            title_y,
            keep_below_min=draw_arguments.keep_min_outside_y_points,
            keep_above_max=draw_arguments.keep_max_outside_y_points,
        )
    # Filter out by max and min value
    if draw_arguments.max_x_value is not None or draw_arguments.min_x_value is not None:
        data_dict = filter_values(data_dict, title_x, draw_arguments.min_x_value, draw_arguments.max_x_value)
    if draw_arguments.max_y_value is not None or draw_arguments.min_y_value is not None:
        data_dict = filter_values(data_dict, title_y, draw_arguments.min_y_value, draw_arguments.max_y_value)
    # Check labels
    model_names, dataset_names = filter_skipped_labels(
        model_names, dataset_names, set(data_dict[key_model_name]), set(data_dict[key_dataset_name])
    )
    # Sorted model names by X position
    model_name_with_x_position = get_sorted_model_names_with_x_position(data_x, model_names)
    model_names_sorted_by_x_position = [model_name for model_name, _ in model_name_with_x_position]

    data = DataFrame(data_dict)
    # Draw lines
    sns.lineplot(
        data=data,
        x=title_x,
        y=title_y,
        hue=key_dataset_name,
        hue_order=dataset_names,
        markers=False,
        legend=False,
        linewidth=1.5,
    )
    # Draw points
    plot = sns.scatterplot(
        data=data,
        x=title_x,
        y=title_y,
        hue=key_dataset_name,
        hue_order=dataset_names,
        style=key_model_name,
        style_order=model_names_sorted_by_x_position,
        # Add black border, "ec" = "edgecolor" and "fc" = "facecolor"
        ec="black",
        legend="full",
    )
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1), ncol=draw_arguments.legend_columns, fontsize="xx-small")

    if draw_arguments.add_linear_regression:
        LOGGER.info("Drawing linear regression for all points")
        sns.regplot(data=data, x=title_x, y=title_y, scatter=False, n_boot=2500, ax=plot, line_kws={"linewidth": 3.0})

    if draw_arguments.add_label_at_top:
        LOGGER.info("Drawing labels at top of plot")
        _, max_y_value = plot.get_ylim()
        draw_label_on_top(plot, model_name_with_x_position, y_position=max_y_value)

    LOGGER.info(f"Saving plot in: {draw_arguments.save_path}")
    figure = plot.get_figure()
    figure.savefig(draw_arguments.save_path, bbox_inches="tight")
