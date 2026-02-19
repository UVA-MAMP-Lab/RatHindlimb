from __future__ import annotations

import argparse
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from osimpy import sto_to_df


def _select_informative_columns(
    columns: list[str],
    values_by_column: dict[str, np.ndarray],
    max_columns: int,
) -> list[str]:
    variances: list[tuple[str, float]] = []
    for column in columns:
        values = values_by_column[column]
        if values.size == 0:
            continue
        variance = float(np.nanvar(values))
        if np.isnan(variance):
            continue
        variances.append((column, variance))

    variances.sort(key=lambda item: item[1], reverse=True)
    if len(variances) == 0:
        return columns[:max_columns]
    return [column for column, _ in variances[:max_columns]]


def _plot_table(
    file_path: Path,
    title: str,
    output_path: Path,
    max_columns: int = 12,
) -> None:
    dataframe, _ = sto_to_df(str(file_path))
    if "time" not in dataframe.columns:
        raise ValueError(f"Missing 'time' column in {file_path}")

    time = dataframe["time"].to_numpy()
    candidate_columns = [column for column in dataframe.columns if column != "time"]
    values_by_column = {
        column: dataframe[column].to_numpy() for column in candidate_columns
    }
    plot_columns = _select_informative_columns(
        columns=candidate_columns,
        values_by_column=values_by_column,
        max_columns=max_columns,
    )

    n_plots = len(plot_columns)
    n_cols = 3
    n_rows = max(1, ceil(n_plots / n_cols))
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(16, 3.5 * n_rows),
        squeeze=False,
        sharex=True,
    )
    flat_axes = axes.flatten()

    for index, column in enumerate(plot_columns):
        axis = flat_axes[index]
        axis.plot(time, values_by_column[column], linewidth=1.2)
        axis.set_title(column, fontsize=9)
        axis.grid(alpha=0.3)

    for index in range(n_plots, len(flat_axes)):
        flat_axes[index].axis("off")

    for axis in flat_axes:
        if axis.has_data():
            axis.set_xlabel("Time (s)")

    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _resolve_cmc_file(session_dir: Path, preferred_name: str) -> Path:
    preferred = session_dir / preferred_name
    if preferred.exists():
        return preferred
    generic = session_dir / f"_{preferred_name.split('_', 1)[-1]}"
    if generic.exists():
        return generic
    raise FileNotFoundError(
        f"Could not find CMC output {preferred_name} or {generic.name} in {session_dir}"
    )


def main(session_directory: str, trial: str) -> list[Path]:
    session_dir = Path(session_directory).resolve()

    ik_file = session_dir / f"{trial}_ik.mot"
    id_file = session_dir / f"{trial}_id.sto"
    cmc_controls_file = _resolve_cmc_file(session_dir, f"{trial}_controls.sto")
    cmc_force_file = _resolve_cmc_file(session_dir, f"{trial}_Actuation_force.sto")

    outputs = [
        session_dir / f"{trial}_kinematics.png",
        session_dir / f"{trial}_dynamics.png",
        session_dir / f"{trial}_cmc_controls.png",
        session_dir / f"{trial}_cmc_actuation_force.png",
    ]

    _plot_table(ik_file, f"{trial} Kinematics (IK)", outputs[0])
    _plot_table(id_file, f"{trial} Dynamics (ID)", outputs[1])
    _plot_table(cmc_controls_file, f"{trial} CMC Controls", outputs[2])
    _plot_table(cmc_force_file, f"{trial} CMC Actuation Force", outputs[3])

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot IK, ID, and CMC OpenSim outputs.")
    parser.add_argument(
        "session_directory",
        type=str,
        help="Session directory containing trial output files.",
    )
    parser.add_argument(
        "--trial",
        type=str,
        default="Walk09",
        help="Trial stem to plot (default: Walk09).",
    )

    cli_args = parser.parse_args()
    generated_files = main(cli_args.session_directory, cli_args.trial)
    for generated_file in generated_files:
        print(generated_file)