import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from matplotlib.patches import Patch

import pandas as pd

import utils


def plot_metric_violins(all_window, metric, early_idx, *,
                        lower_better=None, model_order=None, model_labels=None,
                        colors=('#BDBDBD', '#0072B2'), figsize=(9, 4.3),
                        xlabel=None, ylabel=None, title=None, ylim=None, rotation=None,
                        legand=True, save_path=None, dpi_save=600):

    _LOWER = {'RMSE', 'MAE', 'MSE', 'SMAPE_%', 'MAPE_%', 'DTW_distance',
              'Pearson_distance', 'Cosine_distance'}
    if lower_better is None:
        lower_better = metric in _LOWER or metric.endswith('_distance')

    models = [m for m in (list(all_window.keys()) if model_order is None else list(model_order))
              if model_labels is None or m in model_labels]

    overall, early = [], []
    for m in models:
        v = np.asarray(all_window[m][1][metric], dtype=float)
        overall.append(v[np.isfinite(v)])
        e = v[np.asarray(early_idx)]
        early.append(e[np.isfinite(e)])

    n_all, n_early = len(all_window[models[0]][1][metric]), len(early_idx)

    if model_order is None:
        order = np.argsort([np.median(v) for v in overall])
        if not lower_better:
            order = order[::-1]
        order = list(order)
        if 'current' in models:
            order.remove(models.index('current'))
            order = [models.index('current')] + order
        models = [models[i] for i in order]
        overall = [overall[i] for i in order]
        early = [early[i] for i in order]

    _RC = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none'
    }

    base, off, width = np.arange(len(models)), 0.19, 0.34
    pairs = [
        (overall, -off, colors[0], f'Overall (all {n_all} windows)'),
        (early, +off, colors[1], f'Early (first {n_early} windows)')
    ]

    with mpl.rc_context(_RC):
        fig, ax = plt.subplots(figsize=figsize, dpi=150)

        for i in range(len(models)):
            for series, dpos, color, _ in pairs:

                d, pos = series[i], base[i] + dpos

                if d.size == 0:
                    continue

                if np.ptp(d) == 0:
                    ax.plot([pos-width/3, pos+width/3],
                            [d[0], d[0]],
                            color=color, lw=2.5,
                            solid_capstyle='butt',
                            zorder=2)
                    continue

                vp = ax.violinplot(
                    [d],
                    positions=[pos],
                    widths=width,
                    showextrema=False
                )

                body = vp['bodies'][0]

                verts = body.get_paths()[0].vertices
                x, y = verts[:, 0], verts[:, 1]
                xc = (x.min() + x.max()) / 2
                half_width = np.max(np.abs(x - xc))
                keep = np.abs(x - xc) > 0.02 * half_width

                if np.any(keep):
                    ymin, ymax = y[keep].min(), y[keep].max()
                    verts[:, 1] = np.clip(y, ymin, ymax)

                body.set(
                    facecolor=mpl.colors.to_rgba(color, 0.6),
                    edgecolor=mpl.colors.to_rgba(color, 1.0),
                    linewidth=0.8,
                    zorder=2
                )

                ax.boxplot(
                    [d],
                    positions=[pos],
                    widths=width*0.25,
                    patch_artist=True,
                    showfliers=False,
                    showcaps=False,
                    whis=1.5,
                    medianprops=dict(color='k', lw=1.2, zorder=4),
                    boxprops=dict(facecolor='white',
                                  edgecolor='k',
                                  lw=0.7,
                                  zorder=3),
                    whiskerprops=dict(color='k',
                                      lw=0.7,
                                      zorder=3)
                )

        labels = [(model_labels or {}).get(m, m) for m in models]
        rot = rotation if rotation is not None else 0

        ax.set_xticks(base)
        ax.set_xticklabels(labels, rotation=rot)
        ax.set_xlim(base[0]-0.55, base[-1]+0.55)
        ax.set_ylabel(ylabel or metric.replace('_', ' '))

        if xlabel:
            ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)

        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, ls='--', lw=0.5, alpha=0.5)
        ax.set_axisbelow(True)

        if legand:
            ax.legend(
                handles=[
                    Patch(facecolor=mpl.colors.to_rgba(c, 0.6),
                          edgecolor=mpl.colors.to_rgba(c, 1.0),
                          label=lab)
                    for _, _, c, lab in pairs
                ],
                frameon=False,
                loc='best',
                handlelength=1.4
            )

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path,
                        dpi=dpi_save,
                        bbox_inches='tight')

    return fig, ax


def plot_window_metrics(
    all_window,
    window_metric_names,
    figsize=None,
    cmap_name="tab20",
    marker="o",
    markersize=2,
    linewidth=0.8,
    legend_fontsize=12,
    legend_ncol=None,
    date_format="%Y-%m",
    show=True,
    ylim_robust_iqr=None,
):

    if len(window_metric_names) == 0:
        raise ValueError("window_metric_names 不能为空。")

    if ylim_robust_iqr is not None and ylim_robust_iqr <= 0:
        raise ValueError(
            "ylim_robust_iqr 必须是正数 (箱线图 fence 的 IQR 倍数, 常用 1.5 或 3.0)。"
        )

    print(
        " | ".join(
            f"{label}: {len(values[0])} windows"
            for label, values in all_window.items()
        )
    )

    if figsize is None:
        figsize = (
            12,
            2.4 * len(window_metric_names),
        )

    fig, axes = plt.subplots(
        len(window_metric_names),
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    axes = axes.ravel()
    cmap = plt.get_cmap(cmap_name)

    handles = []
    labels = []

    for li, (label, (time_axis, metrics)) in enumerate(
        all_window.items()
    ):
        color = cmap(li % cmap.N)

        sort_indices = np.argsort(time_axis)
        sorted_time = time_axis[sort_indices]

        for ax, metric_name in zip(
            axes,
            window_metric_names,
        ):
            sorted_metric = (
                metrics[metric_name][sort_indices]
            )

            line = ax.plot(
                sorted_time,
                sorted_metric,
                linestyle="-",
                marker=marker,
                markersize=markersize,
                linewidth=linewidth,
                color=color,
                label=label,
            )[0]

            if metric_name == window_metric_names[0]:
                handles.append(line)
                labels.append(label)

    for ax, metric_name in zip(
        axes,
        window_metric_names,
    ):
        ax.set_ylabel(metric_name)

        ax.grid(
            True,
            linestyle="--",
            alpha=0.4,
        )

    if ylim_robust_iqr is not None:
        k = ylim_robust_iqr
        for ax, metric_name in zip(
            axes,
            window_metric_names,
        ):
            vals = np.concatenate([
                metrics[metric_name]
                for _, metrics in all_window.values()
            ])
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                continue
            lo, hi = q1 - k * iqr, q3 + k * iqr
            pad = 0.05 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)

    axes[-1].set_xlabel(
        "Prediction Start Time"
    )

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter(date_format)
    )

    fig.autofmt_xdate()

    if labels:
        ncol = legend_ncol if legend_ncol is not None else len(labels)
        fig.legend(
            handles,
            labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=ncol,
            fontsize=legend_fontsize,
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    if show:
        plt.show()

    return fig, axes

def plot_best_result_by_time(
    results_sets,
    window_metric_names,
    higher_is_better=None,
    cmap_name="tab20",
    marker="o",
    markersize=5,
    date_format="%Y-%m",
    xlabel="Prediction Start Time",
    title="Best Result Set at Each Prediction Time",
    legend_fontsize=8,
    ylabel_fontsize=10,
    title_fontsize=13,
    grid=True,
    annotate_value=False,
    annotation_fontsize=7,
    show=True,
    figsize=None,
):

    if not isinstance(results_sets, dict) or len(results_sets) == 0:
        raise ValueError("results_sets 必须是非空字典。")

    if len(window_metric_names) == 0:
        raise ValueError("window_metric_names 不能为空。")

    default_higher_is_better = {
        "Pearson_correlation": True,
        "Cosine_similarity": True,
        "R2": True,
        "R2_score": True,
        "Accuracy": True,
        "coverage": True,

        "MSE": False,
        "RMSE": False,
        "MAE": False,
        "MAPE_%": False,
        "SMAPE_%": False,
        "DTW_distance": False,
        "Pearson_distance": False,
        "Cosine_distance": False,
        "pinball_mean": False,
    }

    if higher_is_better is None:
        higher_is_better = {}

    metric_directions = {}

    for metric_name in window_metric_names:
        if metric_name in higher_is_better:
            metric_directions[metric_name] = bool(
                higher_is_better[metric_name]
            )
        elif metric_name in default_higher_is_better:
            metric_directions[metric_name] = (
                default_higher_is_better[metric_name]
            )
        elif metric_name.startswith("coverage"):
            metric_directions[metric_name] = True
        elif metric_name.startswith("pinball"):
            metric_directions[metric_name] = False
        else:
            raise ValueError(
                f"无法判断指标 {metric_name!r} 是越大越好还是越小越好。"
                "\n请通过 higher_is_better 指定，例如："
                f"\n{{'{metric_name}': False}}"
            )

    def compute_window_metrics(res):
        start_time = []
        metrics = {
            name: []
            for name in window_metric_names
        }

        for window_idx, rd in enumerate(res):
            metric_result = utils.evaluate_time_series_similarity(
                rd["target"],
                rd["prediction"],
                verbose=False,
            )

            for metric_name in window_metric_names:
                if metric_name not in metric_result:
                    raise KeyError(
                        f"结果集第 {window_idx} 个窗口中不存在指标 "
                        f"{metric_name!r}。"
                    )

                metrics[metric_name].append(
                    metric_result[metric_name]
                )

            target_index = np.asarray(
                rd["target_index"]
            ).ravel()

            if target_index.size == 0:
                raise ValueError(
                    f"结果集第 {window_idx} 个窗口的 "
                    "target_index 为空。"
                )

            date_array = np.asarray(rd["date"])
            first_target_index = int(target_index[0])

            if (
                first_target_index < 0
                or first_target_index >= len(date_array)
            ):
                raise IndexError(
                    f"第 {window_idx} 个窗口的 target_index[0]="
                    f"{first_target_index} 超出 date 范围。"
                )

            start_time.append(
                pd.Timestamp(date_array[first_target_index])
            )

        start_time = pd.DatetimeIndex(start_time)

        metrics = {
            name: np.asarray(values, dtype=float)
            for name, values in metrics.items()
        }

        return start_time, metrics

    all_window = {
        label: compute_window_metrics(res)
        for label, res in results_sets.items()
    }

    print(
        " | ".join(
            f"{label}: {len(values[0])} windows"
            for label, values in all_window.items()
        )
    )

    result_labels = list(results_sets.keys())

    rows = []

    for label, (time_axis, metrics) in all_window.items():
        for i, time_value in enumerate(time_axis):
            row = {
                "time": pd.Timestamp(time_value),
                "label": label,
            }

            for metric_name in window_metric_names:
                row[metric_name] = metrics[metric_name][i]

            rows.append(row)

    metric_df = pd.DataFrame(rows)

    metric_df = (
        metric_df
        .groupby(["time", "label"], as_index=False)[window_metric_names]
        .mean()
        .sort_values("time")
        .reset_index(drop=True)
    )

    best_results = {}

    for metric_name in window_metric_names:
        metric_rows = []

        for time_value, group in metric_df.groupby("time"):
            values = group[metric_name].to_numpy(dtype=float)
            finite_mask = np.isfinite(values)

            if not np.any(finite_mask):
                continue

            valid_group = group.loc[finite_mask]
            valid_values = values[finite_mask]

            if metric_directions[metric_name]:
                best_position = int(np.argmax(valid_values))
            else:
                best_position = int(np.argmin(valid_values))

            best_row = valid_group.iloc[best_position]

            metric_rows.append({
                "time": pd.Timestamp(time_value),
                "best_label": best_row["label"],
                "best_value": float(best_row[metric_name]),
            })

        best_metric_df = (
            pd.DataFrame(metric_rows)
            .sort_values("time")
            .reset_index(drop=True)
        )

        best_results[metric_name] = {
            "time": best_metric_df["time"].to_numpy(),
            "best_label": best_metric_df["best_label"].to_numpy(),
            "best_value": best_metric_df["best_value"].to_numpy(
                dtype=float
            ),
        }

    if figsize is None:
        figsize = (
            12,
            2.4 * len(window_metric_names),
        )

    fig, axes = plt.subplots(
        len(window_metric_names),
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )

    axes = axes.ravel()

    cmap = plt.get_cmap(cmap_name)

    label_to_y = {
        label: i
        for i, label in enumerate(result_labels)
    }

    label_to_color = {
        label: cmap(i % cmap.N)
        for i, label in enumerate(result_labels)
    }

    for ax, metric_name in zip(
        axes,
        window_metric_names,
    ):
        metric_best = best_results[metric_name]

        times = pd.to_datetime(metric_best["time"])
        best_labels = metric_best["best_label"]
        best_values = metric_best["best_value"]

        y_values = np.asarray([
            label_to_y[label]
            for label in best_labels
        ])

        for label in result_labels:
            selected = best_labels == label

            if not np.any(selected):
                continue

            ax.plot(
                times[selected],
                y_values[selected],
                linestyle="none",
                marker=marker,
                markersize=markersize,
                color=label_to_color[label],
                label=label,
            )

        if annotate_value:
            for time_value, y_value, value in zip(
                times,
                y_values,
                best_values,
            ):
                ax.annotate(
                    f"{value:.3g}",
                    xy=(time_value, y_value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=annotation_fontsize,
                )

        direction_text = (
            "higher is better"
            if metric_directions[metric_name]
            else "lower is better"
        )

        ax.set_ylabel(
            f"{metric_name}\n({direction_text})",
            fontsize=ylabel_fontsize,
        )

        ax.set_yticks(
            np.arange(len(result_labels))
        )

        ax.set_yticklabels(
            result_labels
        )

        ax.set_ylim(
            -0.5,
            len(result_labels) - 0.5,
        )

        if grid:
            ax.grid(
                True,
                axis="x",
                linestyle="--",
                alpha=0.4,
            )

            ax.grid(
                True,
                axis="y",
                linestyle=":",
                alpha=0.25,
            )

    axes[-1].set_xlabel(xlabel)

    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter(date_format)
    )

    if title is not None:
        fig.suptitle(
            title,
            fontsize=title_fontsize,
        )

    legend_handles = [
        plt.Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=markersize,
            color=label_to_color[label],
            label=label,
        )
        for label in result_labels
    ]

    fig.legend(
        handles=legend_handles,
        labels=result_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(result_labels), 6),
        fontsize=legend_fontsize,
        frameon=False,
    )

    fig.autofmt_xdate()
    fig.tight_layout(
        rect=(0, 0.05, 1, 0.97)
    )

    if show:
        plt.show()

    return fig, axes, best_results, all_window
