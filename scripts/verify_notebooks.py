"""Execute the core tutorial notebooks and fail on any runtime error."""

import argparse
import os
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOKS = [
    ROOT / "notebooks" / "part1" / "02_scientific_data_baseline.ipynb",
    ROOT / "notebooks" / "part1" / "04_dynamics_surrogate.ipynb",
    ROOT / "notebooks" / "part1" / "05_sequential_screening.ipynb",
]

REQUIRED_CONTRACT_MARKERS = {
    "02_scientific_data_baseline.ipynb": (
        "CONTRACT scientific-data: models=4 repeats=6 grouped_overlap=0 metrics=mae,rmse"
    ),
    "04_dynamics_surrogate.ipynb": (
        "CONTRACT dynamics: time_split=disjoint transition_models=5 "
        "rollout_regions=2 energy_checked=yes"
    ),
    "05_sequential_screening.ipynb": (
        "CONTRACT screening: strategies=3 repeats=12 budget=20 "
        "ledger_schema=12 unique_queries=yes"
    ),
}

RESULT_EXPECTATIONS = {
    "02_scientific_data_baseline.ipynb": {
        "random_ridge_mean_mae": (0.344, 0.002),
        "grouped_ridge_mean_mae": (1.767, 0.002),
    },
    "04_dynamics_surrogate.ipynb": {
        "rf_interpolation_position_mae": (0.0061, 0.001),
        "rf_interpolation_velocity_mae": (0.0089, 0.001),
        "rf_extrapolation_position_mae": (0.0955, 0.001),
        "rf_extrapolation_velocity_mae": (0.3795, 0.001),
        "zero_extrapolation_velocity_mae": (0.1263, 0.001),
    },
    "05_sequential_screening.ipynb": {
        "random_mean_final_best": (1.818, 0.002),
        "greedy_mean_final_best": (2.070, 0.002),
        "ucb_mean_final_best": (2.055, 0.002),
        "random_success_95pct": (0.167, 0.002),
        "greedy_success_95pct": (0.250, 0.002),
        "ucb_success_95pct": (0.167, 0.002),
    },
}


def parse_result_contract(rendered_text: str, contract_name: str) -> dict[str, float]:
    prefix = f"RESULT_CONTRACT {contract_name} "
    lines = [line for line in rendered_text.splitlines() if line.startswith(prefix)]
    if len(lines) != 1:
        raise RuntimeError(
            f"result contract {contract_name!r}: expected one line, found {len(lines)}"
        )
    values: dict[str, float] = {}
    for token in lines[0][len(prefix) :].split():
        key, raw_value = token.split("=", 1)
        values[key] = float(raw_value)
    return values


def execute_notebook(path: Path, write: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    executed = client.execute()

    error_outputs = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.output_type == "error"
    ]
    if error_outputs:
        raise RuntimeError(f"{path}: {len(error_outputs)} error output(s)")

    executed_code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    if not executed_code_cells or any(cell.execution_count is None for cell in executed_code_cells):
        raise RuntimeError(f"{path}: not every code cell executed")

    rendered_text = "\n".join(
        "".join(output.get("text", []))
        for cell in executed_code_cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    required_marker = REQUIRED_CONTRACT_MARKERS.get(path.name)
    if required_marker and required_marker not in rendered_text:
        raise RuntimeError(f"{path}: missing result contract marker")

    expected_values = RESULT_EXPECTATIONS.get(path.name, {})
    if expected_values:
        contract_name = path.name.split("_", 1)[1].removesuffix(".ipynb")
        if contract_name == "scientific_data_baseline":
            contract_name = "scientific-data"
        elif contract_name == "dynamics_surrogate":
            contract_name = "dynamics"
        elif contract_name == "sequential_screening":
            contract_name = "screening"
        actual_values = parse_result_contract(rendered_text, contract_name)
        if actual_values.keys() != expected_values.keys():
            raise RuntimeError(f"{path}: result contract schema changed")
        for key, (expected, tolerance) in expected_values.items():
            actual = actual_values[key]
            if abs(actual - expected) > tolerance:
                raise RuntimeError(
                    f"{path}: {key}={actual:.6g}, expected {expected:.6g} ± {tolerance:.6g}"
                )

    if write:
        nbformat.write(executed, path)
    print(f"PASS {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebook paths; defaults to the three core tutorials.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist executed outputs back to the notebook files.",
    )
    args = parser.parse_args()

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ai4s-in-hand-matplotlib")
    )
    targets = args.notebooks or DEFAULT_NOTEBOOKS
    for target in targets:
        path = target if target.is_absolute() else ROOT / target
        execute_notebook(path.resolve(), args.write)


if __name__ == "__main__":
    main()
