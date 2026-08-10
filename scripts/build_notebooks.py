"""Build the reader-facing tutorial notebooks from reviewable Python sources."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "part1"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(name: str, cells: list) -> None:
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11+"},
            "ai4s_in_hand": {
                "mode": "tutorial",
                "data_source": "deterministic synthetic data generated in-notebook",
                "license": "CC BY-NC-SA 4.0",
            },
        },
    )
    path = OUT / name
    nbf.write(notebook, path)
    print(f"built {path.relative_to(ROOT)}")


def build_scientific_data_baseline() -> None:
    cells = [
        markdown(
            """
            # 科学数据与可信基线：随机划分为什么会过于乐观

            ## Goal

            用一个可完全复现的“分组科学数据”实验比较随机划分与按科学实体分组的划分。重点不是追求高分，而是检查评价问题是否与目标问题一致。

            对应正文：[第 2 章：科学数据与可信基线](../../docs/part1-ai-foundations/chapter2.md)。
            """
        ),
        markdown(
            """
            ## Setup

            ### Key Assumptions

            - 每个 `scaffold` 模拟一个分子骨架，同一骨架下有多个类似物；
            - 骨架带来未被两个连续描述符完全解释的系统效应；
            - 随机划分允许同一骨架同时出现在训练和测试中，回答“预测已见骨架的新类似物”；
            - 分组划分把完整骨架留作测试，回答“预测未见骨架”。

            数据是教学用合成数据，不代表任何真实化学测量，也不能用于材料或药物决策。
            """
        ),
        code(
            """
            from pathlib import Path
            import platform

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import sklearn
            from sklearn.compose import ColumnTransformer
            from sklearn.dummy import DummyRegressor
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import Ridge
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            from sklearn.model_selection import GroupShuffleSplit, train_test_split
            from sklearn.neural_network import MLPRegressor
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler

            RANDOM_SEED = 20260810
            rng = np.random.default_rng(RANDOM_SEED)

            print({
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "seed": RANDOM_SEED,
            })
            """
        ),
        markdown("## Steps\n\n### 1. Generate grouped scientific observations"),
        code(
            """
            n_scaffolds = 60
            analogues_per_scaffold = 8
            scaffold_names = np.array([f"S{i:02d}" for i in range(n_scaffolds)])
            scaffold_effect = rng.normal(0.0, 2.0, size=n_scaffolds)

            rows = []
            for scaffold_index, scaffold in enumerate(scaffold_names):
                for analogue in range(analogues_per_scaffold):
                    descriptor_1 = rng.normal()
                    descriptor_2 = rng.uniform(-1.0, 1.0)
                    measurement_noise = rng.normal(0.0, 0.25)
                    target = (
                        1.8 * descriptor_1
                        - 1.1 * descriptor_2
                        + scaffold_effect[scaffold_index]
                        + measurement_noise
                    )
                    rows.append(
                        {
                            "scaffold": scaffold,
                            "analogue": analogue,
                            "descriptor_1": descriptor_1,
                            "descriptor_2": descriptor_2,
                            "target": target,
                        }
                    )

            data = pd.DataFrame(rows)
            print(f"rows={len(data)}, scaffolds={data['scaffold'].nunique()}")
            data.head()
            """
        ),
        markdown("### 2. Audit grain, missingness, duplicates, and group sizes"),
        code(
            """
            audit = {
                "rows": len(data),
                "unique_scaffold_analogue_pairs": data[["scaffold", "analogue"]].drop_duplicates().shape[0],
                "missing_cells": int(data.isna().sum().sum()),
                "min_group_size": int(data.groupby("scaffold").size().min()),
                "max_group_size": int(data.groupby("scaffold").size().max()),
            }
            assert audit["rows"] == audit["unique_scaffold_analogue_pairs"]
            assert audit["missing_cells"] == 0
            print(audit)
            """
        ),
        markdown("### 3. Build a model ladder with the same preprocessing boundary"),
        code(
            """
            feature_columns = ["scaffold", "descriptor_1", "descriptor_2"]
            numeric_columns = ["descriptor_1", "descriptor_2"]
            categorical_columns = ["scaffold"]

            def make_preprocessing():
                return ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), numeric_columns),
                        (
                            "scaffold",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            categorical_columns,
                        ),
                    ]
                )

            def make_models(seed):
                return {
                    "mean": DummyRegressor(strategy="mean"),
                    "ridge": make_pipeline(make_preprocessing(), Ridge(alpha=1.0)),
                    "forest": make_pipeline(
                        make_preprocessing(),
                        RandomForestRegressor(
                            n_estimators=160,
                            min_samples_leaf=3,
                            random_state=seed,
                            n_jobs=1,
                        ),
                    ),
                    "mlp": make_pipeline(
                        make_preprocessing(),
                        MLPRegressor(
                            hidden_layer_sizes=(20,),
                            activation="tanh",
                            solver="adam",
                            alpha=0.1,
                            max_iter=2000,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=30,
                            learning_rate_init=0.01,
                            tol=1e-4,
                            random_state=seed,
                        ),
                    ),
                }

            def evaluate_split(train_indices, test_indices, split_name, repeat):
                train = data.iloc[train_indices]
                test = data.iloc[test_indices]
                shared_groups = set(train["scaffold"]) & set(test["scaffold"])
                rows = []
                prediction_rows = []
                for model_name, model in make_models(RANDOM_SEED + repeat).items():
                    model.fit(train[feature_columns], train["target"])
                    prediction = model.predict(test[feature_columns])
                    rows.append(
                        {
                            "repeat": repeat,
                            "split": split_name,
                            "model": model_name,
                            "train_rows": len(train),
                            "test_rows": len(test),
                            "shared_scaffolds": len(shared_groups),
                            "mae": mean_absolute_error(test["target"], prediction),
                            "rmse": mean_squared_error(test["target"], prediction) ** 0.5,
                        }
                    )
                    if repeat == 0:
                        prediction_rows.extend(
                            {
                                "split": split_name,
                                "model": model_name,
                                "scaffold": scaffold,
                                "observed": observed,
                                "predicted": predicted,
                            }
                            for scaffold, observed, predicted in zip(
                                test["scaffold"], test["target"], prediction
                            )
                        )
                return rows, prediction_rows
            """
        ),
        markdown("### 4. Repeat random and scaffold-held-out evaluation under matched budgets"),
        code(
            """
            all_indices = np.arange(len(data))
            n_repeats = 6
            result_rows = []
            prediction_rows = []
            for repeat in range(n_repeats):
                split_seed = RANDOM_SEED + repeat
                random_train, random_test = train_test_split(
                    all_indices, test_size=0.25, random_state=split_seed
                )
                group_splitter = GroupShuffleSplit(
                    n_splits=1, test_size=0.25, random_state=split_seed
                )
                grouped_train, grouped_test = next(
                    group_splitter.split(data, groups=data["scaffold"])
                )
                for split_name, train_indices, test_indices in [
                    ("Random row split", random_train, random_test),
                    ("Held-out scaffold split", grouped_train, grouped_test),
                ]:
                    rows, predictions = evaluate_split(
                        train_indices, test_indices, split_name, repeat
                    )
                    result_rows.extend(rows)
                    prediction_rows.extend(predictions)

            results = pd.DataFrame(result_rows)
            predictions = pd.DataFrame(prediction_rows)
            assert (results.loc[results["split"] == "Held-out scaffold split", "shared_scaffolds"] == 0).all()
            summary = (
                results.groupby(["split", "model"])[["mae", "rmse"]]
                .agg(["mean", "std"])
                .round(3)
            )
            summary
            """
        ),
        code(
            """
            plot_data = results.groupby(["split", "model"])["mae"].mean().unstack()
            ax = plot_data.plot(
                kind="bar", figsize=(9, 4), color=["#9ca3af", "#f59e0b", "#2563eb", "#7c3aed"]
            )
            ax.set_title("The same model answers different questions under different splits")
            ax.set_ylabel("MAE (synthetic target units; lower is better)")
            ax.set_xlabel("")
            ax.set_ylim(bottom=0)
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.show()

            ridge_predictions = predictions[predictions["model"] == "ridge"]
            fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
            for axis, (split_name, frame) in zip(axes, ridge_predictions.groupby("split")):
                group_codes = pd.factorize(frame["scaffold"])[0]
                axis.scatter(
                    frame["observed"], frame["predicted"], c=group_codes,
                    cmap="tab20", alpha=0.62, s=24
                )
                limits = [
                    min(frame["observed"].min(), frame["predicted"].min()),
                    max(frame["observed"].max(), frame["predicted"].max()),
                ]
                axis.plot(limits, limits, color="#dc2626", linestyle=":")
                axis.set_title(split_name)
                axis.set_xlabel("Observed")
            axes[0].set_ylabel("Predicted")
            fig.suptitle("Ridge diagnostics; color encodes test scaffold")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## Checks

            - 分组划分的训练与测试骨架交集必须为零；
            - 两种划分使用同一目标、特征、模型、重复数和测试比例；
            - 同时报告 MAE、RMSE 与重复间离散度，不把两种问题伪装成同一总体；
            - 如果随机划分明显更好，合理解释是它可复用已见骨架效应，而不是模型普遍更懂化学。
            """
        ),
        code(
            """
            assert set(results["model"]) == {"mean", "ridge", "forest", "mlp"}
            assert results.groupby(["split", "model"]).size().eq(n_repeats).all()
            assert np.isfinite(results[["mae", "rmse"]].to_numpy()).all()
            ridge_means = results[results["model"] == "ridge"].groupby("split")["mae"].mean()
            print(f"随机划分 Ridge 平均 MAE: {ridge_means['Random row split']:.3f}")
            print(f"骨架分组 Ridge 平均 MAE: {ridge_means['Held-out scaffold split']:.3f}")
            print(
                "RESULT_CONTRACT scientific-data "
                f"random_ridge_mean_mae={ridge_means['Random row split']:.6f} "
                f"grouped_ridge_mean_mae={ridge_means['Held-out scaffold split']:.6f}"
            )
            print("CONTRACT scientific-data: models=4 repeats=6 grouped_overlap=0 metrics=mae,rmse")
            print("本次确定性合成实验的差值不应外推为真实分子数据的固定幅度。")
            """
        ),
        markdown(
            """
            ## Next Steps

            1. 删除 `scaffold` 特征，判断差距来自实体记忆还是连续描述符；
            2. 改变骨架效应和噪声，观察结论在哪些条件下消失；
            3. 将 `scaffold` 替换为时间、实验批次或空间区域，重新解释分组划分；
            4. 在真实数据上使用官方定义的分组规则，并同时记录重复、缺失和标签条件。

            **结论边界：**这个实验展示了在当前数据生成机制下，随机划分回答了更容易的问题；它不证明分组划分总是唯一正确，也不量化任何真实数据集的偏差。
            """
        ),
    ]
    write_notebook("02_scientific_data_baseline.ipynb", cells)


def build_dynamics_surrogate() -> None:
    cells = [
        markdown(
            """
            # 动力系统代理模型：插值准确不等于外推可信

            ## Goal

            用阻尼振子的确定性轨迹训练一个轻量代理模型，分别检查训练时间范围内的插值、范围外外推、动力学残差和推理时间。

            对应正文：[第 4 章：科学机器学习与计算加速](../../docs/part1-ai-foundations/chapter4.md)。
            """
        ),
        markdown(
            r"""
            ## Setup

            ### Key Assumptions

            状态为位移 $x$ 与速度 $v$，满足

            $$\dot{x}=v,\qquad \dot{v}=-2\zeta\omega v-\omega^2x.$$

            使用无量纲量，参数固定为 $\omega=1.4$、$\zeta=0.12$。代理只看到 $t\leq 8$ 的样本；$t>8$ 专门用于外推检查。解析轨迹只是教学真值，并不代表真实昂贵模拟器。
            """
        ),
        code(
            """
            from time import perf_counter
            import platform

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import sklearn
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_absolute_error
            from sklearn.neural_network import MLPRegressor
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            RANDOM_SEED = 20260810
            OMEGA = 1.4
            ZETA = 0.12
            TRAIN_END = 8.0
            END_TIME = 16.0

            print({
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "seed": RANDOM_SEED,
            })
            """
        ),
        markdown("## Steps\n\n### 1. Generate the reference trajectory"),
        code(
            """
            def reference_state(time_points, initial_state=(1.0, 0.0)):
                time_points = np.asarray(time_points)
                decay = ZETA * OMEGA
                damped_frequency = OMEGA * np.sqrt(1.0 - ZETA**2)
                initial_position, initial_velocity = initial_state
                coefficient = (initial_velocity + decay * initial_position) / damped_frequency
                exponential = np.exp(-decay * time_points)
                position = exponential * (
                    initial_position * np.cos(damped_frequency * time_points)
                    + coefficient * np.sin(damped_frequency * time_points)
                )
                velocity = exponential * (
                    -decay * (
                        initial_position * np.cos(damped_frequency * time_points)
                        + coefficient * np.sin(damped_frequency * time_points)
                    )
                    - initial_position * damped_frequency * np.sin(damped_frequency * time_points)
                    + coefficient * damped_frequency * np.cos(damped_frequency * time_points)
                )
                return np.column_stack([position, velocity])

            full_time = np.linspace(0.0, END_TIME, 801)
            full_state = reference_state(full_time)
            assert np.allclose(full_state[0], [1.0, 0.0])
            """
        ),
        markdown("### 2. Create a held-out interpolation set and a true extrapolation set"),
        code(
            """
            training_mask = full_time <= TRAIN_END
            training_candidates = np.flatnonzero(training_mask)
            train_indices = training_candidates[training_candidates % 5 != 0]
            interpolation_indices = training_candidates[training_candidates % 5 == 0]
            extrapolation_indices = np.flatnonzero(full_time > TRAIN_END)

            model = RandomForestRegressor(
                n_estimators=200,
                min_samples_leaf=2,
                random_state=RANDOM_SEED,
                n_jobs=1,
            )
            model.fit(full_time[train_indices, None], full_state[train_indices])

            interpolation_prediction = model.predict(full_time[interpolation_indices, None])
            extrapolation_prediction = model.predict(full_time[extrapolation_indices, None])
            zero_baseline = np.zeros_like(extrapolation_prediction)

            metrics = pd.DataFrame(
                [
                    {
                        "region": "训练范围内插值",
                        "model": "随机森林代理",
                        "position_mae": mean_absolute_error(full_state[interpolation_indices, 0], interpolation_prediction[:, 0]),
                        "velocity_mae": mean_absolute_error(full_state[interpolation_indices, 1], interpolation_prediction[:, 1]),
                    },
                    {
                        "region": "训练范围外外推",
                        "model": "随机森林代理",
                        "position_mae": mean_absolute_error(full_state[extrapolation_indices, 0], extrapolation_prediction[:, 0]),
                        "velocity_mae": mean_absolute_error(full_state[extrapolation_indices, 1], extrapolation_prediction[:, 1]),
                    },
                    {
                        "region": "训练范围外外推",
                        "model": "零平衡基线",
                        "position_mae": mean_absolute_error(full_state[extrapolation_indices, 0], zero_baseline[:, 0]),
                        "velocity_mae": mean_absolute_error(full_state[extrapolation_indices, 1], zero_baseline[:, 1]),
                    },
                ]
            )
            metrics.round(4)
            """
        ),
        markdown("### 3. Inspect trajectories rather than only average error"),
        code(
            """
            full_prediction = model.predict(full_time[:, None])
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            labels = [(0, "Position x"), (1, "Velocity v")]
            for axis, (column, label) in zip(axes, labels):
                axis.plot(full_time, full_state[:, column], color="#111827", label="Reference")
                axis.plot(full_time, full_prediction[:, column], color="#2563eb", linestyle="--", label="Surrogate")
                axis.axvline(TRAIN_END, color="#dc2626", linestyle=":", label="Training boundary" if column == 0 else None)
                axis.set_ylabel(label)
                axis.grid(alpha=0.2)
            axes[0].legend(ncol=3)
            axes[-1].set_xlabel("Dimensionless time t")
            fig.suptitle("Interpolation inside the training range and boundary behavior outside")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown("### 4. Compute equation residuals"),
        code(
            """
            time_step = full_time[1] - full_time[0]
            predicted_dx_dt = np.gradient(full_prediction[:, 0], time_step)
            predicted_dv_dt = np.gradient(full_prediction[:, 1], time_step)
            residual_x = predicted_dx_dt - full_prediction[:, 1]
            residual_v = (
                predicted_dv_dt
                + 2.0 * ZETA * OMEGA * full_prediction[:, 1]
                + OMEGA**2 * full_prediction[:, 0]
            )

            residual_report = pd.DataFrame(
                {
                    "region": ["训练范围", "外推范围"],
                    "mean_abs_dx_minus_v": [
                        np.mean(np.abs(residual_x[training_mask])),
                        np.mean(np.abs(residual_x[~training_mask])),
                    ],
                    "mean_abs_dv_equation": [
                        np.mean(np.abs(residual_v[training_mask])),
                        np.mean(np.abs(residual_v[~training_mask])),
                    ],
                }
            )
            residual_report.round(4)
            """
        ),
        markdown("### 5. Learn the actual one-step state transition"),
        code(
            """
            PAIR_DT = 0.04
            pair_time = np.arange(0.0, TRAIN_END + PAIR_DT / 2, PAIR_DT)
            training_initial_states = np.array(
                [
                    [1.0, 0.0], [0.7, 0.4], [-0.8, 0.2], [0.3, -0.9],
                    [1.2, -0.3], [-0.4, -0.7], [0.6, 0.8], [-1.0, 0.5],
                ]
            )
            pair_inputs = []
            pair_targets = []
            for initial_state in training_initial_states:
                states = reference_state(pair_time, initial_state)
                pair_inputs.append(states[:-1])
                pair_targets.append(states[1:])
            pair_inputs = np.vstack(pair_inputs)
            pair_targets = np.vstack(pair_targets)

            linear_transition = LinearRegression(fit_intercept=False)
            linear_transition.fit(pair_inputs, pair_targets)
            mlp_transitions = []
            for seed_offset in range(3):
                mlp = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(16,),
                        activation="tanh",
                        solver="lbfgs",
                        alpha=1e-4,
                        max_iter=800,
                        random_state=RANDOM_SEED + seed_offset,
                    ),
                )
                mlp.fit(pair_inputs, pair_targets)
                mlp_transitions.append(mlp)

            rollout_time = np.arange(0.0, END_TIME + PAIR_DT / 2, PAIR_DT)
            test_initial_state = np.array([0.65, -0.35])
            rollout_reference = reference_state(rollout_time, test_initial_state)

            def rollout(predict_next):
                states = [test_initial_state.copy()]
                for _ in range(len(rollout_time) - 1):
                    states.append(np.asarray(predict_next(states[-1])).reshape(2))
                return np.vstack(states)

            transition_rollouts = {
                "persistence": rollout(lambda state: state),
                "linear": rollout(lambda state: linear_transition.predict(np.asarray(state)[None, :])[0]),
            }
            for seed_offset, mlp in enumerate(mlp_transitions):
                transition_rollouts[f"mlp_seed_{seed_offset}"] = rollout(
                    lambda state, fitted=mlp: fitted.predict(np.asarray(state)[None, :])[0]
                )
            """
        ),
        markdown("### 6. Check rolling error and energy, not only one-step loss"),
        code(
            """
            def mechanical_energy(states):
                return 0.5 * states[:, 1] ** 2 + 0.5 * OMEGA**2 * states[:, 0] ** 2

            transition_rows = []
            interpolation_rollout_mask = (rollout_time > 0) & (rollout_time <= TRAIN_END)
            extrapolation_rollout_mask = rollout_time > TRAIN_END
            for model_name, predicted_states in transition_rollouts.items():
                energy = mechanical_energy(predicted_states)
                energy_difference = np.diff(energy)
                for region, mask in [
                    ("rollout_to_training_horizon", interpolation_rollout_mask),
                    ("rollout_beyond_training_horizon", extrapolation_rollout_mask),
                ]:
                    region_energy_difference = energy_difference[mask[1:]]
                    transition_rows.append(
                        {
                            "model": model_name,
                            "region": region,
                            "position_mae": mean_absolute_error(
                                rollout_reference[mask, 0], predicted_states[mask, 0]
                            ),
                            "velocity_mae": mean_absolute_error(
                                rollout_reference[mask, 1], predicted_states[mask, 1]
                            ),
                            "energy_increase_count": int(np.sum(region_energy_difference > 1e-6)),
                            "max_energy_increase": float(
                                np.maximum(region_energy_difference, 0).max(initial=0.0)
                            ),
                        }
                    )
            transition_metrics = pd.DataFrame(transition_rows)
            transition_metrics.round(5)
            """
        ),
        code(
            """
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            for column, axis, label in [(0, axes[0], "Position x"), (1, axes[1], "Velocity v")]:
                axis.plot(rollout_time, rollout_reference[:, column], color="#111827", label="Reference")
                axis.plot(rollout_time, transition_rollouts["persistence"][:, column], color="#9ca3af", linestyle=":", label="Persistence")
                axis.plot(rollout_time, transition_rollouts["linear"][:, column], color="#2563eb", linestyle="--", label="Linear transition")
                axis.plot(rollout_time, transition_rollouts["mlp_seed_0"][:, column], color="#7c3aed", alpha=0.8, label="MLP (seed 0)")
                axis.axvline(TRAIN_END, color="#dc2626", linestyle=":")
                axis.set_ylabel(label)
                axis.grid(alpha=0.2)
            axes[0].legend(ncol=4, fontsize=8)
            axes[-1].set_xlabel("Dimensionless time t")
            fig.suptitle("Rolling a learned state transition beyond the training horizon")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown("### 7. Time a bounded batch (illustrative, not a universal benchmark)"),
        code(
            """
            query_time = np.linspace(0.0, END_TIME, 20_000)
            start = perf_counter()
            _ = reference_state(query_time)
            reference_seconds = perf_counter() - start

            start = perf_counter()
            _ = model.predict(query_time[:, None])
            surrogate_seconds = perf_counter() - start

            timing = {
                "queries": len(query_time),
                "analytic_reference_seconds": reference_seconds,
                "surrogate_seconds": surrogate_seconds,
            }
            print(timing)
            print("解析函数本来就很便宜；该计时不能用于声称代理加速了真实数值模拟。")
            """
        ),
        markdown(
            """
            ## Checks

            - 插值与外推按时间边界定义，不能随机混在一个测试集；
            - 代理与零基线使用同一外推样本；
            - 方程残差与状态误差是不同指标，二者都不能单独证明模型可信；
            - 计时只描述本机本次运行，且解析参考并不昂贵。
            """
        ),
        code(
            """
            assert full_time[train_indices].max() <= TRAIN_END
            assert full_time[extrapolation_indices].min() > TRAIN_END
            assert np.isfinite(metrics[["position_mae", "velocity_mae"]].to_numpy()).all()
            assert np.isfinite(residual_report.iloc[:, 1:].to_numpy()).all()
            assert set(transition_metrics["model"]) == {
                "persistence", "linear", "mlp_seed_0", "mlp_seed_1", "mlp_seed_2"
            }
            assert transition_metrics.groupby("model").size().eq(2).all()
            assert np.isfinite(
                transition_metrics[["position_mae", "velocity_mae", "max_energy_increase"]].to_numpy()
            ).all()
            reference_energy = mechanical_energy(rollout_reference)
            assert np.all(np.diff(reference_energy) <= 1e-10)
            time_curve_lookup = metrics.set_index(["region", "model"])
            print(
                "RESULT_CONTRACT dynamics "
                f"rf_interpolation_position_mae={time_curve_lookup.loc[('训练范围内插值', '随机森林代理'), 'position_mae']:.6f} "
                f"rf_interpolation_velocity_mae={time_curve_lookup.loc[('训练范围内插值', '随机森林代理'), 'velocity_mae']:.6f} "
                f"rf_extrapolation_position_mae={time_curve_lookup.loc[('训练范围外外推', '随机森林代理'), 'position_mae']:.6f} "
                f"rf_extrapolation_velocity_mae={time_curve_lookup.loc[('训练范围外外推', '随机森林代理'), 'velocity_mae']:.6f} "
                f"zero_extrapolation_velocity_mae={time_curve_lookup.loc[('训练范围外外推', '零平衡基线'), 'velocity_mae']:.6f}"
            )
            print("CONTRACT dynamics: time_split=disjoint transition_models=5 rollout_regions=2 energy_checked=yes")
            print("数据边界、误差、滚动和能量检查通过。")
            """
        ),
        markdown(
            """
            ## Next Steps

            1. 改变阻尼和频率，把参数作为输入，测试参数空间外推；
            2. 用可微神经网络替代树模型，在训练损失中加入方程残差；
            3. 改变滚动步长或加入一步噪声，观察误差如何积累；
            4. 在昂贵模拟器上重新设计公平的精度—速度基准。

            **结论边界：**本实验说明插值分数不能代表外推可信性；它不证明随机森林不适合所有动力系统，也不证明加入物理损失必然改善所有指标。
            """
        ),
    ]
    write_notebook("04_dynamics_surrogate.ipynb", cells)


def build_sequential_screening() -> None:
    cells = [
        markdown(
            """
            # 从预测到决策：有限预算下的序贯筛选

            ## Goal

            在一个确定性虚拟候选池中比较随机选择、预测均值贪心和不确定性感知 UCB。评价对象是固定查询预算下的 best-so-far 轨迹，而不是一次静态测试分数。

            对应正文：[第 5 章：从预测到决策与发现](../../docs/part1-ai-foundations/chapter5.md)。
            """
        ),
        markdown(
            """
            ## Setup

            ### Key Assumptions

            - 候选池和真实目标在实验开始前固定，但算法只能查询被选择候选的标签；
            - 三种策略使用相同初始样本、预算和随机种子；
            - UCB 分数为预测均值加探索系数乘标准差，仅是一个教学策略；
            - 结果只适用于当前目标函数、模型、候选池和预算，不说明某策略普遍最优。
            """
        ),
        code(
            """
            from time import perf_counter
            import platform

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import sklearn
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

            BASE_SEED = 20260810
            N_CANDIDATES = 240
            INITIAL_SAMPLES = 6
            QUERY_BUDGET = 20
            N_REPEATS = 12

            print({
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "base_seed": BASE_SEED,
            })
            """
        ),
        markdown("## Steps\n\n### 1. Build a virtual candidate pool"),
        code(
            """
            pool_rng = np.random.default_rng(BASE_SEED)
            candidates = pool_rng.uniform(-2.5, 2.5, size=(N_CANDIDATES, 2))

            def virtual_objective(points):
                x_1 = points[:, 0]
                x_2 = points[:, 1]
                broad_peak = 1.8 * np.exp(-0.7 * ((x_1 - 0.9) ** 2 + (x_2 + 0.6) ** 2))
                narrow_peak = 2.6 * np.exp(-2.8 * ((x_1 + 1.2) ** 2 + (x_2 - 1.0) ** 2))
                background = 0.25 * np.sin(2.2 * x_1) * np.cos(1.7 * x_2)
                return broad_peak + narrow_peak + background

            true_values = virtual_objective(candidates)
            global_best = float(true_values.max())
            print({"candidates": len(candidates), "hidden_global_best": round(global_best, 4)})
            """
        ),
        markdown("### 2. Define equal-budget strategies"),
        code(
            """
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)

            def make_model():
                return GaussianProcessRegressor(
                    kernel=kernel,
                    optimizer=None,
                    normalize_y=True,
                    random_state=BASE_SEED,
                )

            def run_strategy(strategy, seed):
                rng = np.random.default_rng(seed)
                selected = list(rng.choice(N_CANDIDATES, size=INITIAL_SAMPLES, replace=False))
                best_history = [float(true_values[selected].max())]
                trace_rows = [
                    {
                        "phase": "initial",
                        "query_step": 0,
                        "candidate_id": int(candidate_id),
                        "predicted_mean": np.nan,
                        "predicted_std": np.nan,
                        "acquisition": np.nan,
                        "feedback": float(true_values[candidate_id]),
                        "elapsed_seconds": 0.0,
                        "failure_status": "ok",
                    }
                    for candidate_id in selected
                ]

                for query_step in range(1, QUERY_BUDGET + 1):
                    step_start = perf_counter()
                    available = np.setdiff1d(np.arange(N_CANDIDATES), selected, assume_unique=False)

                    if strategy == "random":
                        next_index = int(rng.choice(available))
                        selected_mean = np.nan
                        selected_std = np.nan
                        selected_acquisition = np.nan
                    else:
                        model = make_model()
                        model.fit(candidates[selected], true_values[selected])
                        mean, std = model.predict(candidates[available], return_std=True)
                        if strategy == "greedy":
                            acquisition = mean
                        elif strategy == "ucb":
                            acquisition = mean + 1.5 * std
                        else:
                            raise ValueError(f"unknown strategy: {strategy}")
                        best_available_position = int(np.argmax(acquisition))
                        next_index = int(available[best_available_position])
                        selected_mean = float(mean[best_available_position])
                        selected_std = float(std[best_available_position])
                        selected_acquisition = float(acquisition[best_available_position])

                    selected.append(next_index)
                    best_history.append(float(true_values[selected].max()))
                    trace_rows.append(
                        {
                            "phase": "query",
                            "query_step": query_step,
                            "candidate_id": next_index,
                            "predicted_mean": selected_mean,
                            "predicted_std": selected_std,
                            "acquisition": selected_acquisition,
                            "feedback": float(true_values[next_index]),
                            "elapsed_seconds": perf_counter() - step_start,
                            "failure_status": "ok",
                        }
                    )

                assert len(selected) == len(set(selected))
                assert len(selected) == INITIAL_SAMPLES + QUERY_BUDGET
                assert np.all(np.diff(best_history) >= -1e-12)
                return np.array(best_history), np.array(selected), pd.DataFrame(trace_rows)
            """
        ),
        markdown("### 3. Repeat strategies under matched initial conditions"),
        code(
            """
            strategy_names = ["random", "greedy", "ucb"]
            histories = {strategy: [] for strategy in strategy_names}
            trace_frames = []

            for repeat in range(N_REPEATS):
                matched_seed = BASE_SEED + repeat
                for strategy in strategy_names:
                    history, _, trace = run_strategy(strategy, matched_seed)
                    histories[strategy].append(history)
                    trace.insert(0, "seed", matched_seed)
                    trace.insert(0, "repeat", repeat)
                    trace.insert(0, "strategy", strategy)
                    trace_frames.append(trace)

            histories = {name: np.vstack(values) for name, values in histories.items()}
            decision_trace = pd.concat(trace_frames, ignore_index=True)
            summary_rows = []
            for strategy, values in histories.items():
                final_values = values[:, -1]
                summary_rows.append(
                    {
                        "strategy": strategy,
                        "mean_final_best": final_values.mean(),
                        "median_final_best": np.median(final_values),
                        "success_within_95pct_of_pool_best": np.mean(final_values >= 0.95 * global_best),
                    }
                )
            summary = pd.DataFrame(summary_rows)
            summary.round(3)
            """
        ),
        markdown("### 4. Inspect the auditable decision ledger"),
        code(
            """
            trace_columns = [
                "strategy", "repeat", "seed", "phase", "query_step", "candidate_id",
                "predicted_mean", "predicted_std", "acquisition", "feedback",
                "elapsed_seconds", "failure_status",
            ]
            decision_trace[trace_columns].head(12)
            """
        ),
        markdown("### 5. Plot the decision metric with repeat variability"),
        code(
            """
            colors = {"random": "#6b7280", "greedy": "#dc2626", "ucb": "#2563eb"}
            labels = {"random": "Random", "greedy": "Mean-greedy", "ucb": "UCB"}
            steps = np.arange(QUERY_BUDGET + 1)

            fig, ax = plt.subplots(figsize=(9, 5))
            for strategy in strategy_names:
                values = histories[strategy]
                mean = values.mean(axis=0)
                lower = np.quantile(values, 0.25, axis=0)
                upper = np.quantile(values, 0.75, axis=0)
                ax.plot(steps, mean, label=labels[strategy], color=colors[strategy])
                ax.fill_between(steps, lower, upper, color=colors[strategy], alpha=0.15)

            ax.axhline(global_best, color="#111827", linestyle=":", label="Pool optimum (evaluation only)")
            ax.set_title(f"Best-so-far under a fixed budget ({N_REPEATS} matched repeats)")
            ax.set_xlabel("Additional queries")
            ax.set_ylabel("Best observed objective")
            ax.legend()
            ax.grid(alpha=0.2)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown("### 6. Inspect one run in candidate space"),
        code(
            """
            example_history, example_selected, example_trace = run_strategy("ucb", BASE_SEED)
            fig, ax = plt.subplots(figsize=(7, 6))
            scatter = ax.scatter(
                candidates[:, 0], candidates[:, 1], c=true_values, cmap="viridis", s=28, alpha=0.65
            )
            ax.plot(
                candidates[example_selected, 0], candidates[example_selected, 1],
                color="white", linewidth=1.0, alpha=0.8
            )
            ax.scatter(
                candidates[example_selected, 0], candidates[example_selected, 1],
                facecolors="none", edgecolors="#dc2626", s=80, label="UCB queries"
            )
            ax.set_title("One UCB run (colors are visible only during evaluation)")
            ax.set_xlabel("Candidate descriptor 1")
            ax.set_ylabel("Candidate descriptor 2")
            ax.legend()
            fig.colorbar(scatter, ax=ax, label="Hidden objective")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## Checks

            - 每次运行不重复查询候选；
            - 三种策略使用相同初始样本、总预算和候选池；
            - best-so-far 轨迹只能不降；
            - 全局最优标签只用于事后评价，不能泄漏给选择策略；
            - 图中阴影是重复运行的四分位区间，不是不确定性的通用置信区间。
            """
        ),
        code(
            """
            for strategy, values in histories.items():
                assert values.shape == (N_REPEATS, QUERY_BUDGET + 1)
                assert np.all(np.diff(values, axis=1) >= -1e-12)
                assert np.all(values <= global_best + 1e-12)
            assert list(decision_trace.columns) == trace_columns
            expected_trace_rows = len(strategy_names) * N_REPEATS * (INITIAL_SAMPLES + QUERY_BUDGET)
            assert len(decision_trace) == expected_trace_rows
            assert decision_trace.groupby(["strategy", "repeat"])["candidate_id"].nunique().eq(
                INITIAL_SAMPLES + QUERY_BUDGET
            ).all()
            assert decision_trace.loc[decision_trace["phase"] == "query", "query_step"].between(
                1, QUERY_BUDGET
            ).all()
            assert decision_trace["failure_status"].eq("ok").all()
            summary_lookup = summary.set_index("strategy")
            print(
                "RESULT_CONTRACT screening "
                f"random_mean_final_best={summary_lookup.loc['random', 'mean_final_best']:.6f} "
                f"greedy_mean_final_best={summary_lookup.loc['greedy', 'mean_final_best']:.6f} "
                f"ucb_mean_final_best={summary_lookup.loc['ucb', 'mean_final_best']:.6f} "
                f"random_success_95pct={summary_lookup.loc['random', 'success_within_95pct_of_pool_best']:.6f} "
                f"greedy_success_95pct={summary_lookup.loc['greedy', 'success_within_95pct_of_pool_best']:.6f} "
                f"ucb_success_95pct={summary_lookup.loc['ucb', 'success_within_95pct_of_pool_best']:.6f}"
            )
            print("CONTRACT screening: strategies=3 repeats=12 budget=20 ledger_schema=12 unique_queries=yes")
            print("预算、唯一查询、轨迹单调性、上界和审计账本检查通过。")
            """
        ),
        markdown(
            """
            ## Next Steps

            1. 改变探索系数，观察 UCB 的探索—利用权衡；
            2. 加入查询成本和不可行候选，改写为约束优化；
            3. 用校准较差的模型重复实验，观察不确定性误导决策的方式；
            4. 在真实材料或反应数据上先明确“一个查询”代表计算、测量还是实验。

            **结论边界：**某策略在本候选池上取得更高平均 best-so-far，只说明它在声明的模型、预算和目标下表现更好；不能推出它在所有科学优化问题中占优，更不能把找到高值候选直接称为科学发现。
            """
        ),
    ]
    write_notebook("05_sequential_screening.ipynb", cells)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_scientific_data_baseline()
    build_dynamics_surrogate()
    build_sequential_screening()


if __name__ == "__main__":
    main()
