"""Plotting helpers for manuscript figures."""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402

from jpeg_defense import frequency  # noqa: E402


IEEE_RCPARAMS = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.15,
    "savefig.facecolor": "white",
}

SCHEDULE_STYLES = {
    "front_loaded": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "fixed": {"color": "#D55E00", "linestyle": "--", "marker": "s"},
    "geometric": {"color": "#009E73", "linestyle": "-.", "marker": "^"},
    "arithmetic": {"color": "#666666", "linestyle": ":", "marker": "D"},
}

SCHEDULE_HATCHES = {
    "front_loaded": "////",
    "fixed": "\\\\\\\\",
    "geometric": "xx",
    "arithmetic": "..",
}

SCHEDULE_ORDER = tuple(SCHEDULE_STYLES)

plt.rcParams.update(IEEE_RCPARAMS)


def load_metrics_csv(path):
    """Load experiment metrics from a CSV path."""
    return pd.read_csv(Path(path))


def make_asr_by_generation(metrics_csv, out_dir):
    """Write Fig. 1: CIFAR-10 ResNet-18 PGD R1 ASR by JPEG generation."""
    df = _filter_cifar_rows(load_metrics_csv(metrics_csv))
    focused = _c10_resnet18_pgd_r1_subset(df)
    if not focused.empty:
        df = focused
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "figure1_asr_by_generation.pdf"
    png_path = out_dir / "figure1_asr_by_generation.png"

    generation_col = _generation_column(df)
    _require_columns(df, ["schedule", generation_col, "asr"])

    panels = _plot_panels(df)
    fig, axes = plt.subplots(
        *_panel_grid_shape(len(panels)),
        figsize=_panel_grid_figsize(len(panels), width_per_col=3.45, height_per_row=2.25),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax, (title, panel) in zip(flat_axes, panels):
        handles = []
        labels = []
        for schedule in _ordered_schedules(panel["schedule"].drop_duplicates()):
            group = panel[panel["schedule"] == schedule]
            grouped = group.groupby(generation_col, as_index=False)["asr"].mean()
            style = _schedule_style(schedule)
            (line,) = ax.plot(
                grouped[generation_col],
                grouped["asr"],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.1,
                markerfacecolor="white",
                markeredgewidth=0.8,
                linewidth=1.25,
                label=_schedule_display_name(schedule),
            )
            handles.append(line)
            labels.append(_schedule_display_name(schedule))

        ax.set_xlabel("Generation")
        ax.set_title(_figure1_panel_title(panel, title), pad=1.0)
        ax.set_ylim(*_padded_ylim(panel["asr"], lower=0.0, upper=1.0, min_span=0.16))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_xticks(sorted(panel[generation_col].drop_duplicates()))
        ax.grid(True, alpha=0.28, linewidth=0.45)
        _legend_above(ax, handles, labels, ncol=min(4, max(1, len(handles))), y=1.14)
    for ax in flat_axes[len(panels):]:
        ax.axis("off")
    axes[0][0].set_ylabel("Attack success rate")
    fig.tight_layout(pad=0.35)
    _save_figure(fig, pdf_path, png_path)
    return pdf_path, png_path


def make_estimator_scatter(metrics_csv, out_dir):
    """Write estimator scatter outputs, falling back to an ASR proxy when needed."""
    df = load_metrics_csv(metrics_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "figure2_estimator_scatter.pdf"
    png_path = out_dir / "figure2_estimator_scatter.png"

    _require_columns(df, ["asr"])
    x_col, y_col, x_label, y_label = _estimator_columns(df)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(df[x_col], df[y_col], s=46, alpha=0.85)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Estimator agreement")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, pdf_path, png_path)
    return pdf_path, png_path


def make_first_qf_effect(metrics_csv, out_dir):
    """Write first-stage quality factor versus ASR figure outputs."""
    df = load_metrics_csv(metrics_csv).copy()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "figure2_first_qf_effect.pdf"
    png_path = out_dir / "figure2_first_qf_effect.png"

    _require_columns(df, ["qfs", "asr", "schedule"])
    df["first_qf"] = df["qfs"].map(_first_qf)

    panels = _plot_panels(df)
    fig, axes = plt.subplots(
        *_panel_grid_shape(len(panels)),
        figsize=_panel_grid_figsize(len(panels), width_per_col=2.2, height_per_row=2.0),
        sharey=True,
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax, (title, panel) in zip(flat_axes, panels):
        for schedule, group in panel.groupby("schedule", sort=True):
            ax.scatter(
                group["first_qf"],
                group["asr"],
                s=30,
                alpha=0.82,
                label=_schedule_display_name(schedule),
            )
        ax.set_xlabel("First-stage JPEG QF")
        ax.set_title(str(title), fontsize=8)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, alpha=0.3)
    for ax in flat_axes[len(panels):]:
        ax.axis("off")
    axes[0][0].set_ylabel("Attack success rate")
    flat_axes[min(len(panels), len(flat_axes)) - 1].legend(title="Schedule", loc="best")
    fig.tight_layout()
    _save_figure(fig, pdf_path, png_path)
    return pdf_path, png_path


def make_frequency_matching_figure(frequency_csv, out_dir):
    """Write perturbation-threshold diagnostic figure outputs."""
    df = _filter_cifar_rows(load_metrics_csv(frequency_csv).copy())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "figure2_frequency_matching.pdf"
    png_path = out_dir / "figure2_frequency_matching.png"

    _require_columns(
        df,
        [
            "dataset",
            "model",
            "attack",
            "omega_delta",
            "predicted_best_schedule",
        ],
    )
    if "tau" not in df.columns:
        df["tau"] = df["dataset"].map(frequency.elimination_threshold_for_dataset)
    if "omega_delta_relation" not in df.columns:
        df["omega_delta_relation"] = df.apply(
            lambda row: frequency.omega_delta_relation(row["omega_delta"], row["tau"]),
            axis=1,
        )
    jpeg_aware_attack = _jpeg_aware_series(df)
    df["predicted_best_schedule"] = [
        frequency.predict_schedule_from_omega_delta(omega_delta, tau, jpeg_aware)
        for omega_delta, tau, jpeg_aware in zip(
            df["omega_delta"],
            df["tau"],
            jpeg_aware_attack,
        )
    ]
    df["label"] = df.apply(
        lambda row: _condition_display_label(
            f"{row['dataset']} {row['model']} {row['attack']}"
        ),
        axis=1,
    )
    df = df.sort_values(["dataset", "model", "attack"]).reset_index(drop=True)
    df["threshold_margin"] = df["omega_delta"].astype(float) - df["tau"].astype(float)
    y_positions = list(range(len(df)))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, max(3.4, 0.38 * len(df) + 1.3)),
        sharey=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    axes[0].barh(
        y_positions,
        df["omega_delta"],
        height=0.44,
        label=r"$\omega_\delta$",
        color="#4C78A8",
    )
    for position, row in enumerate(df.itertuples(index=False)):
        axes[0].plot(
            [float(row.tau), float(row.tau)],
            [position - 0.30, position + 0.30],
            color="#222222",
            linewidth=1.0,
        )
        axes[0].text(
            float(row.tau) + 0.03,
            position + 0.25,
            r"$\tau$",
            va="center",
            ha="left",
            fontsize=6.2,
            color="#222222",
        )
    axes[0].set_xlabel("DCT radial centroid")
    axes[0].set_yticks(y_positions, labels=df["label"])
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="x", alpha=0.3)
    axes[0].legend(loc="lower right")

    colors = [
        "#54A24B" if str(value).strip() == ">" else "#E45756"
        for value in df["omega_delta_relation"]
    ]
    axes[1].barh(y_positions, df["threshold_margin"], color=colors, height=0.44)
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel(r"$\omega_\delta-\tau$")
    axes[1].grid(True, axis="x", alpha=0.3)
    for position, row in enumerate(df.itertuples(index=False)):
        x_value = float(row.threshold_margin)
        offset = 0.04 if x_value >= 0 else -0.04
        horizontal_alignment = "left" if x_value >= 0 else "right"
        axes[1].text(
            x_value + offset,
            position,
            _schedule_display_name(row.predicted_best_schedule),
            va="center",
            ha=horizontal_alignment,
            fontsize=6.5,
        )

    fig.suptitle("Perturbation-threshold schedule diagnostics")
    fig.tight_layout()
    _save_figure(fig, pdf_path, png_path)
    return pdf_path, png_path


def make_centroid_vs_delta_scatter(frequency_csv, source_data_csv, out_dir):
    """Write Fig. 2: signed centroid coordinate versus paired ASR difference."""
    frequency_df = _normalize_frequency_df(load_metrics_csv(frequency_csv).copy())
    source_df = _filter_cifar_rows(load_metrics_csv(source_data_csv).copy())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "figure2_centroid_vs_delta.pdf"
    png_path = out_dir / "figure2_centroid_vs_delta.png"

    condition_cols = _condition_key_columns(frequency_df, source_df)
    _require_columns(frequency_df, [*condition_cols, "omega_delta", "tau"])
    _require_columns(source_df, [*condition_cols, "schedule", "attack_success"])

    margins = frequency_df[[*condition_cols, "omega_delta", "tau"]].copy()
    margins["signed_margin"] = (
        margins["omega_delta"].astype(float) - margins["tau"].astype(float)
    )
    margins = margins.drop_duplicates(condition_cols)

    paired = _paired_asr_delta_pp(source_df, condition_cols)
    plot_df = margins.merge(paired, on=condition_cols, how="inner")
    if plot_df.empty:
        raise ValueError("no paired front_loaded/fixed ASR rows match frequency metrics")
    plot_df["label"] = plot_df.apply(
        lambda row: _condition_display_label(_condition_join(row, condition_cols)),
        axis=1,
    )
    plot_df = plot_df.sort_values(condition_cols).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(3.55, max(3.1, 0.17 * len(plot_df) + 2.35)))
    # visual reference: signed centroid margin near the tau threshold.
    ax.axvspan(
        -0.3,
        0.3,
        facecolor="#E6E6E6",
        edgecolor="#B0B0B0",
        hatch="////",
        linewidth=0.0,
        alpha=0.55,
        label=r"$|\omega_\delta-\tau|\leq 0.3$",
    )
    # practical equivalence: paired ASR difference below one percentage point.
    ax.axhspan(
        -1.0,
        1.0,
        facecolor="#F4F4F4",
        edgecolor="#8F8F8F",
        hatch="\\\\",
        linewidth=0.0,
        alpha=0.70,
        label=r"$|\Delta|<1$ pp",
    )
    ax.axvline(0.0, color="black", linewidth=0.85, zorder=2)
    ax.axhline(0.0, color="black", linewidth=0.85, zorder=2)
    ax.scatter(
        plot_df["signed_margin"],
        plot_df["delta_pp"],
        s=48,
        color="#0072B2",
        edgecolor="#222222",
        linewidth=0.55,
        zorder=3,
    )
    sorted_for_labels = plot_df.sort_values(["signed_margin", "delta_pp"]).reset_index()
    for rank, row in enumerate(sorted_for_labels.itertuples(index=False)):
        offset = _annotation_offset(rank, len(sorted_for_labels), row.signed_margin)
        ax.annotate(
            row.label,
            (float(row.signed_margin), float(row.delta_pp)),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.6,
            ha="left" if offset[0] >= 0 else "right",
            va="bottom" if offset[1] >= 0 else "top",
            arrowprops={
                "arrowstyle": "-",
                "color": "#555555",
                "linewidth": 0.35,
                "shrinkA": 0,
                "shrinkB": 2,
            },
            zorder=4,
        )
    ax.set_xlabel(r"Signed centroid coordinate ($\omega_\delta-\tau$)")
    ax.set_ylabel(r"$\Delta$ ASR, front-loaded minus fixed (pp)")
    ax.set_xlim(*_padded_xlim(plot_df["signed_margin"], include=(-0.3, 0.3), min_span=0.9))
    y_low, y_high = _padded_ylim(
        plot_df["delta_pp"],
        lower=None,
        upper=None,
        min_span=6.0,
        include=(-1.0, 1.0),
    )
    ax.set_ylim(y_low, y_high + 1.2)
    ax.grid(True, alpha=0.25, linewidth=0.45)
    _legend_above(ax, ncol=1)
    fig.tight_layout(pad=0.35)
    _save_figure(fig, pdf_path, png_path)
    return pdf_path, png_path


def make_attack_range_summary(metrics_csv, out_dir):
    """Write Fig. 3: range-level lowest-ASR schedule-count summary."""
    df = _filter_cifar_rows(load_metrics_csv(metrics_csv).copy())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "figure3_attack_range_summary.pdf"
    png_path = out_dir / "figure3_attack_range_summary.png"

    grouped = _best_schedule_counts(df)
    pivot = grouped.pivot_table(
        index="label", columns="schedule", values="count", aggfunc="sum", fill_value=0
    ).sort_index()

    schedule_order = [
        schedule for schedule in ("front_loaded", "fixed", "geometric", "arithmetic")
        if schedule in pivot.columns
    ]
    schedule_order.extend(
        schedule for schedule in pivot.columns if schedule not in schedule_order
    )
    pivot = pivot[schedule_order]
    pivot = pivot.rename(columns=_schedule_display_name)
    pivot.index = [_condition_display_label(label) for label in pivot.index]

    fig, ax = plt.subplots(figsize=(3.55, max(2.5, 0.30 * len(pivot.index) + 1.15)))
    y_positions = list(range(len(pivot.index)))
    left = [0.0] * len(pivot.index)
    handles = []
    labels = []
    renamed_to_raw = {_schedule_display_name(schedule): schedule for schedule in schedule_order}
    for display_name in pivot.columns:
        raw_name = renamed_to_raw.get(display_name, display_name.replace("-", "_"))
        values = pivot[display_name].astype(float).to_list()
        style = _schedule_style(raw_name)
        bars = ax.barh(
            y_positions,
            values,
            left=left,
            height=0.66,
            color=style["color"],
            edgecolor="#222222",
            linewidth=0.45,
            hatch=SCHEDULE_HATCHES.get(raw_name, ""),
            label=display_name,
        )
        left = [base + value for base, value in zip(left, values)]
        handles.append(bars[0])
        labels.append(display_name)
    ax.set_xlabel("Lowest-ASR QF-range count")
    ax.set_ylabel("")
    ax.set_yticks(y_positions, labels=pivot.index)
    ax.set_xlim(left=0.0, right=max(left) + 0.25 if left else 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.45)
    ax.invert_yaxis()
    _legend_above(ax, handles, labels, ncol=min(2, max(1, len(handles))))
    fig.tight_layout(pad=0.35)
    _save_figure(fig, pdf_path, png_path)
    return pdf_path, png_path


def _generation_column(df):
    if "generation" in df.columns:
        return "generation"
    if "generations" in df.columns:
        return "generations"
    return "generation"


def _asr_generation_panel_count(df):
    return len(_plot_panels(df))


def _attack_panels(df):
    if "attack" not in df.columns:
        return ["all attacks"]
    attacks = [str(value) for value in df["attack"].drop_duplicates()]
    return attacks or ["all attacks"]


def _plot_panels(df):
    group_cols = _model_group_columns(df)
    if not group_cols:
        return [("all metrics", df)]
    panels = []
    for key, group in df.groupby(group_cols, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        title = " / ".join(str(value) for value in key_tuple)
        panels.append((title, group))
    return panels or [("all metrics", df)]


def _panel_grid_shape(count, max_cols=4):
    count = max(1, int(count))
    cols = min(int(max_cols), count)
    rows = int(math.ceil(count / cols))
    return rows, cols


def _panel_grid_figsize(count, width_per_col, height_per_row):
    rows, cols = _panel_grid_shape(count)
    return max(3.45, width_per_col * cols), max(2.55, height_per_row * rows)


def _ordered_schedules(schedules):
    values = [str(schedule) for schedule in schedules]
    ordered = [schedule for schedule in SCHEDULE_ORDER if schedule in values]
    ordered.extend(sorted(schedule for schedule in values if schedule not in ordered))
    return ordered


def _schedule_style(schedule):
    fallback = {"color": "#333333", "linestyle": "-", "marker": "o"}
    return {**fallback, **SCHEDULE_STYLES.get(str(schedule), {})}


def _legend_above(ax, handles=None, labels=None, ncol=4, y=1.03):
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=max(1, int(ncol)),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.2,
        columnspacing=1.0,
        handletextpad=0.45,
    )


def _figure1_panel_title(panel, fallback):
    parts = []
    for column in ("dataset", "model", "attack", "range_name"):
        if column in panel.columns:
            values = [str(value) for value in panel[column].dropna().unique()]
            if len(values) == 1:
                parts.append(values[0])
    title = _condition_display_label(" ".join(parts)) if parts else str(fallback)
    return title.replace(" R1", ", R1")


def _padded_xlim(values, include=(), min_span=1.0):
    return _padded_limits(values, min_span=min_span, include=include)


def _padded_ylim(values, lower=None, upper=None, min_span=1.0, include=()):
    low, high = _padded_limits(values, min_span=min_span, include=include)
    if lower is not None:
        low = max(float(lower), low)
    if upper is not None:
        high = min(float(upper), high)
    return low, high


def _padded_limits(values, min_span=1.0, include=()):
    series = pd.Series(values).astype(float)
    candidates = list(series.dropna()) + [float(value) for value in include]
    if not candidates:
        return 0.0, float(min_span)
    low = min(candidates)
    high = max(candidates)
    span = high - low
    if span < min_span:
        midpoint = (low + high) / 2.0
        low = midpoint - min_span / 2.0
        high = midpoint + min_span / 2.0
        span = min_span
    pad = max(span * 0.10, min_span * 0.04)
    return low - pad, high + pad


def _annotation_offset(rank, count, signed_margin):
    vertical_cycle = [12, -14, 24, -26, 36, -38]
    y_offset = vertical_cycle[rank % len(vertical_cycle)]
    midpoint = max(1, count // 2)
    margin = float(signed_margin)
    if margin < -0.18:
        x_offset = 13
    elif margin > 1.2:
        x_offset = -16
    elif abs(margin) <= 0.3:
        x_offset = 14 if rank % 2 == 0 else 18
    else:
        x_offset = 9 if rank < midpoint else -9
    return x_offset, y_offset


def _best_schedule_counts(df):
    df = _filter_cifar_rows(df)
    generation_col = _generation_column(df)
    _require_columns(df, ["attack", "range_name", "schedule", generation_col, "asr"])
    group_cols = [*_model_group_columns(df), "range_name"]
    max_generation = df.groupby(group_cols)[generation_col].transform("max")
    subset = df[df[generation_col] == max_generation]
    best = subset.loc[subset.groupby(group_cols)["asr"].idxmin()].copy()
    best["label"] = best.apply(_model_attack_label, axis=1)
    return (
        best.groupby(["label", "schedule"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["label", "schedule"])
    )


def _model_group_columns(df):
    if "model" in df.columns:
        columns = []
        if "dataset" in df.columns:
            columns.append("dataset")
        columns.append("model")
        if "attack" in df.columns:
            columns.append("attack")
        return columns
    if "attack" in df.columns:
        return ["attack"]
    return []


def _summary_label(row):
    parts = []
    for column in ("dataset", "model", "attack", "range_name"):
        if column in row.index:
            parts.append(str(row[column]))
    return " ".join(parts)


def _model_attack_label(row):
    parts = []
    for column in ("dataset", "model", "attack"):
        if column in row.index:
            parts.append(str(row[column]))
    return " ".join(parts)


def _schedule_display_name(schedule):
    names = {
        "front_loaded": "front-loaded",
        "geometric": "geometric",
        "arithmetic": "arithmetic",
        "fixed": "fixed",
    }
    return names.get(str(schedule), str(schedule).replace("_", "-"))


def _condition_display_label(label):
    aliases = {
        "cifar10": "C10",
        "imagenet": "IN",
        "cifar_resnet18": "ResNet-18",
        "robustbench_engstrom2019": "Engstrom19",
        "robustbench_rice2020": "Rice20",
        "robustbench_wong2020fast": "Wong20",
        "imagenet_deit_tiny": "DeiT-Tiny",
        "imagenet_swin_t": "Swin-T",
        "imagenet_vit_b_16": "ViT-B/16",
        "fgsm": "FGSM",
        "pgd": "PGD",
        "jpeg_aware_pgd": "JPEG-aware PGD",
    }
    parts = []
    for token in str(label).split():
        parts.append(aliases.get(token, token.replace("_", "-")))
    return " ".join(parts)


def _first_qf(qfs):
    first = str(qfs).split(",", maxsplit=1)[0].strip().strip('"')
    return int(first)


def _estimator_columns(df):
    candidates = [
        ("estimator_asr", "asr"),
        ("estimated_asr", "asr"),
        ("surrogate_asr", "asr"),
    ]
    for x_col, y_col in candidates:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col, "Estimated ASR", "Observed ASR"
    return "asr", "asr", "Observed ASR (proxy)", "Observed ASR"


def _require_columns(df, columns):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"metrics CSV is missing required columns: {', '.join(missing)}")


def _normalize_frequency_df(df):
    df = _filter_cifar_rows(df)
    if "tau" not in df.columns and "dataset" in df.columns:
        df["tau"] = df["dataset"].map(frequency.elimination_threshold_for_dataset)
    return df


def _condition_key_columns(*frames):
    candidates = ["dataset", "model", "attack"]
    columns = [
        column
        for column in candidates
        if all(column in frame.columns for frame in frames)
    ]
    if not columns:
        raise ValueError("CSV inputs need at least one shared condition column")
    return columns


def _paired_asr_delta_pp(df, condition_cols):
    grouped_cols = [*condition_cols, "schedule"]
    asr = (
        df.groupby(grouped_cols)["attack_success"]
        .mean()
        .reset_index()
        .pivot_table(
            index=condition_cols,
            columns="schedule",
            values="attack_success",
            aggfunc="mean",
        )
    )
    required = ["front_loaded", "fixed"]
    missing = [column for column in required if column not in asr.columns]
    if missing:
        raise ValueError(
            "source data CSV is missing paired schedules: " + ", ".join(missing)
        )
    delta = (asr["front_loaded"] - asr["fixed"]) * 100.0
    return delta.rename("delta_pp").reset_index()


def _condition_join(row, columns):
    return " ".join(str(row[column]) for column in columns)


def _jpeg_aware_series(df):
    if "jpeg_aware_attack" in df.columns:
        return df["jpeg_aware_attack"].map(_truthy)
    if "adaptive_attack" in df.columns:
        return df["adaptive_attack"].map(_truthy)
    return df.get("attack", "").astype(str).str.contains("jpeg_aware", case=False)


def _truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _filter_cifar_rows(df):
    if "dataset" not in df.columns:
        return df
    return df[df["dataset"].astype(str).str.lower() == "cifar10"].copy()


def _c10_resnet18_pgd_r1_subset(df):
    if not {"dataset", "model", "attack", "range_name"}.issubset(df.columns):
        return df.iloc[0:0].copy()
    mask = (
        df["dataset"].astype(str).str.lower().eq("cifar10")
        & df["model"].astype(str).eq("cifar_resnet18")
        & df["attack"].astype(str).str.lower().eq("pgd")
        & df["range_name"].astype(str).eq("R1")
    )
    return df[mask].copy()


def _save_figure(fig, pdf_path, png_path):
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
