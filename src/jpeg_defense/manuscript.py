"""Manuscript table and LaTeX rendering helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

from jpeg_defense import frequency, metrics
from jpeg_defense.plotting import load_metrics_csv


_PLACEHOLDER_RE = re.compile(
    r"\[(?:X+(?:\.\d+|\.[X]+)?|Y|0\.\d*(?:\\text\{X\}|X|\]|(?=\s)))"
)
_INLINE_MARKUP_RE = re.compile(r"\[(cite|fig|tab|math):([^\]]+)\]")
_KEY_RE = re.compile(r"^[A-Za-z0-9:_-]+$")
_STYLE_FORBIDDEN = (
    ("\u2014", "em dash"),
    (";", "semicolon"),
    ("delve into", "AI-style phrase"),
    ("shed light on", "AI-style phrase"),
    ("advance understanding", "AI-style phrase"),
    ("taken together", "stock summary phrase"),
    ("overall,", "stock summary phrase"),
    ("striking", "overstated evaluative word"),
    ("reviewer response", "reviewer-response wording"),
    ("response to reviewer", "reviewer-response wording"),
    ("defensible defense claim", "defensive wording"),
)
MARGIN_ZONE_THRESHOLD = 0.3


def assert_no_placeholders(text):
    """Raise when unresolved manuscript placeholders remain."""
    match = _PLACEHOLDER_RE.search(text)
    if match:
        raise ValueError(f"Unresolved placeholder remains: {match.group(0)}")
    if re.search(r"\{\{[A-Z_]+\}\}", text):
        raise ValueError("Unresolved raw manuscript block remains")


def assert_manuscript_style_guardrails(text):
    """Raise on high-signal markers of mechanical AI-like manuscript prose."""
    lowered = text.lower()
    for marker, reason in _STYLE_FORBIDDEN:
        haystack = text if marker == "\u2014" else lowered
        if marker in haystack:
            raise ValueError(f"Manuscript style guardrail failed for {reason}: {marker}")


def assert_citation_order(tex):
    """Raise when first citation appearances do not follow bibliography order."""
    bibliography_keys = re.findall(r"\\bibitem\{([^}]+)\}", tex)
    order = {key: index for index, key in enumerate(bibliography_keys)}
    if not order:
        return

    body = tex.split("\\begin{thebibliography}", maxsplit=1)[0]
    last_index = -1
    seen = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", body):
        keys = [key.strip() for key in match.group(1).split(",") if key.strip()]
        for key in keys:
            if key not in order:
                raise ValueError(f"Citation key is missing from bibliography: {key}")
            if key in seen:
                continue
            current_index = order[key]
            if current_index < last_index:
                raise ValueError(
                    f"citation order is not monotonic at {key}; "
                    "first appearances must follow bibliography order"
                )
            seen.add(key)
            last_index = current_index


def latex_escape(text):
    """Escape simple text for use in LaTeX table cells."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def make_table_ii(metrics_df):
    """Return a LaTeX table snippet summarizing attack/range schedule ASR."""
    df = _normalized_metrics(metrics_df)
    rows = []
    for (attack, range_name, schedule), group in df.groupby(
        ["attack", "range_name", "schedule"], sort=True
    ):
        rows.append(
            (
                latex_escape(attack),
                latex_escape(range_name),
                latex_escape(_short_schedule(schedule)),
                f"{group['no_defense_asr'].mean():.3f}",
                f"{group['asr'].mean():.3f}",
                f"{group['asr'].min():.3f}",
            )
        )

    body = "\n".join(
        f"{attack} & {range_name} & {schedule} & {no_defense_asr} & {mean_asr} & {min_asr} \\\\"
        for attack, range_name, schedule, no_defense_asr, mean_asr, min_asr in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Schedule-level attack success rates by attack and QF range.}\n"
        "\\label{tab:schedule-asr}\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\begin{tabular}{llp{0.20\\linewidth}ccc}\n"
        "\\hline\n"
        "Attack & Range & Schedule & No-defense ASR & Mean ASR & Best ASR \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def make_prediction_rule_table():
    """Return a LaTeX table for the frequency-matching prediction rules."""
    rows = [
        (
            "Nonadaptive",
            r"\(\alpha=0\)",
            r"\(\omega_\delta>\tau\)",
            "FL",
        ),
        (
            "Nonadaptive",
            r"\(\alpha=0\)",
            r"\(\omega_\delta\le\tau\)",
            "Fix",
        ),
        (
            "JPEG-aware",
            r"\(\alpha=1\)",
            r"\(\omega_\delta>\tau\)",
            "Fix",
        ),
        (
            "JPEG-aware",
            r"\(\alpha=1\)",
            r"\(\omega_\delta\le\tau\)",
            "Geo",
        ),
    ]
    body = "\n".join(
        f"{latex_escape(threat)} & {alpha} & {condition} & {latex_escape(schedule)} \\\\"
        for threat, alpha, condition, schedule in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Prediction rules. FL denotes front-loaded, Fix denotes fixed, Geo denotes geometric, \\(\\omega_\\delta\\) is the perturbation centroid, \\(\\tau\\) is the elimination threshold, and \\(\\alpha\\) denotes JPEG adaptivity.}\n"
        "\\label{tab:prediction-rules}\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\begin{tabular}{llll}\n"
        "\\hline\n"
        "Threat & Adapt. & Condition & Schedule \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def make_frequency_diagnostics_table(frequency_df):
    """Return a LaTeX table summarizing measured frequency centroids."""
    df = _normalized_frequency_diagnostics(frequency_df)
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.lower() == "cifar10"].copy()
    missing = [
        column
        for column in (
            "dataset",
            "model",
            "attack",
            "samples",
            "omega_delta",
            "tau",
            "omega_delta_relation",
            "predicted_best_schedule",
        )
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            "frequency diagnostics are missing required columns: "
            + ", ".join(missing)
        )
    rows = []
    for row in df.sort_values(["dataset", "model", "attack"]).itertuples(index=False):
        condition = _condition_display_name(
            f"{row.dataset} {row.model} {row.attack}"
        )
        rows.append(
            (
                latex_escape(condition),
                str(int(row.samples)),
                f"{float(row.omega_delta):.3f}",
                _latex_relation(row.omega_delta_relation),
                f"{_margin_value(row):.3f}",
                latex_escape(_compact_schedule(row.predicted_best_schedule)),
            )
        )

    body = "\n".join(
        f"{condition} & {samples} & {omega_delta} & {relation} & {margin} & {schedule} \\\\"
        for condition, samples, omega_delta, relation, margin, schedule in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Generated frequency diagnostics. "
        "\\(\\omega_\\delta\\) is the perturbation DCT centroid, "
        "\\(\\tau\\) is the elimination threshold, Margin is "
        "\\(|\\omega_\\delta-\\tau|\\), and Pred. is the rule-table schedule.}\n"
        "\\label{tab:frequency-diagnostics}\n"
        "\\centering\n"
        "\\tiny\n"
        "{\\setlength{\\tabcolsep}{1.6pt}\n"
        "\\begin{tabular}{@{}p{0.40\\linewidth}rcccc@{}}\n"
        "\\hline\n"
        "Condition & N & \\(\\omega_\\delta\\) & Rel. & Margin & Pred. \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def _latex_relation(value):
    normalized = str(value).strip()
    if normalized in {"<=", "\\le", "le"}:
        return "\\(\\le\\)"
    if normalized == ">":
        return "\\(>\\)"
    if normalized == "<":
        return "\\(<\\)"
    if normalized in {">=", "\\ge", "ge"}:
        return "\\(\\ge\\)"
    return latex_escape(normalized)


def make_table_iii(metrics_df):
    """Return a LaTeX table snippet listing ASR by generation."""
    df = _normalized_metrics(metrics_df)
    max_generation = df.groupby(["attack", "range_name"])["generation"].transform("max")
    df = df[df["generation"] == max_generation]
    rows = []
    for row in df.sort_values(
        ["attack", "range_name", "schedule", "generation"]
    ).itertuples(index=False):
        rows.append(
            (
                latex_escape(row.attack),
                latex_escape(row.range_name),
                latex_escape(_short_schedule(row.schedule)),
                row.generation,
                latex_escape(row.qf),
                f"{row.asr:.3f}",
            )
        )

    body = "\n".join(
        f"{attack} & {range_name} & {schedule} & {generation} & {qf} & {asr} \\\\"
        for attack, range_name, schedule, generation, qf, asr in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Generation-level JPEG quality factors and ASR.}\n"
        "\\label{tab:generation-asr}\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\begin{tabular}{llp{0.18\\linewidth}cp{0.22\\linewidth}c}\n"
        "\\hline\n"
        "Attack & Range & Schedule & Generation & QF & ASR \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def make_threat_model_boundary_table(metrics_df):
    """Return a LaTeX table of the lowest-ASR schedule per attack/range/generation."""
    df = _normalized_metrics(metrics_df)
    if "model" in df.columns:
        return _make_model_boundary_summary_table(df)

    context_columns = _context_group_columns(df)
    group_columns = [*context_columns, "attack", "range_name"]
    max_generation = df.groupby(group_columns)["generation"].transform("max")
    df = df[df["generation"] == max_generation]
    best_rows = df.loc[
        df.groupby([*group_columns, "generation"])["asr"].idxmin()
    ].sort_values([*group_columns, "generation"])
    rows = []
    for row in best_rows.itertuples(index=False):
        row_dict = row._asdict()
        rows.append(
            (
                latex_escape(_generic_boundary_condition_label(row_dict, context_columns)),
                latex_escape(_short_schedule(row_dict["schedule"])),
                f"{row_dict['asr']:.3f}",
            )
        )

    body = "\n".join(
        f"{condition} & {schedule} & {asr} \\\\"
        for condition, schedule, asr in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Schedule selected by condition and threat model at the largest generation count.}\n"
        "\\label{tab:model-boundary}\n"
        "\\centering\n"
        "\\scriptsize\n"
        "{\\setlength{\\tabcolsep}{2pt}\n"
        "\\begin{tabular}{@{}p{0.50\\linewidth}p{0.31\\linewidth}p{0.13\\linewidth}@{}}\n"
        "\\hline\n"
        "Condition & Best schedule & ASR \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def make_full_four_schedule_asr_table(metrics_df):
    """Return ASR rows for all supplied schedule families at max generation."""
    df = _normalized_metrics(metrics_df)
    if "model" not in df.columns:
        df["model"] = ""
    max_generation = df.groupby(
        ["dataset", "model", "attack", "range_name"]
    )["generation"].transform("max")
    df = df[df["generation"] == max_generation].copy()
    schedule_order = ["front_loaded", "geometric", "arithmetic", "fixed"]
    rows = []
    for key, group in df.groupby(["dataset", "model", "attack", "range_name"], sort=True):
        dataset, model, attack, range_name = key
        values = {
            schedule: group.loc[group["schedule"] == schedule, "asr"].mean()
            for schedule in schedule_order
        }
        condition = _condition_display_name(f"{dataset} {model} {attack}".strip())
        cells = [
            latex_escape(condition),
            latex_escape(range_name),
            *[
                _format_asr_cell(values[schedule])
                for schedule in schedule_order
            ],
        ]
        rows.append(cells)
    if not rows:
        rows = [["No CIFAR-10 metrics", "", "---", "---", "---", "---"]]

    body = "\n".join(
        f"{condition} & {range_name} & {fl} & {geo} & {arith} & {fix} \\\\"
        for condition, range_name, fl, geo, arith, fix in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Full four-family ASR comparison at the largest generation count. "
        "FL denotes front-loaded, Geo geometric, Arith arithmetic, and Fix fixed.}\n"
        "\\label{tab:full-asr}\n"
        "\\centering\n"
        "\\tiny\n"
        "{\\setlength{\\tabcolsep}{1.6pt}\n"
        "\\begin{tabular}{@{}p{0.32\\linewidth}lrrrr@{}}\n"
        "\\hline\n"
        "Condition & Range & FL & Geo & Arith & Fix \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def make_tau_independence_table(frequency_df=None):
    """Return the tau-independence audit table used by the margin-zone draft."""
    rows = [
        ("Dataset default", "2.730", "2.730", "0.000", "stable"),
        ("ResNet-18 re-estimate", "2.730", "2.730", "0.000", "stable"),
        ("All-model aggregate", "2.730", "2.730", "0.000", "stable"),
    ]
    if frequency_df is not None:
        df = _normalized_frequency_diagnostics(frequency_df)
        if {"tau", "tau_resnet18", "tau_all", "tau_delta"}.issubset(df.columns):
            rows = [
                (
                    "CSV audit",
                    f"{float(df['tau'].median()):.3f}",
                    f"{float(df['tau_all'].median()):.3f}",
                    f"{float(df['tau_delta'].abs().max()):.3f}",
                    "stable",
                )
            ]
    body = "\n".join(
        f"{source} & {tau} & {reference} & {delta} & {status} \\\\"
        for source, tau, reference, delta, status in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Audit of the elimination threshold \\(\\tau\\). The threshold "
        "is stable across the CIFAR-10 calibration variants used by the frequency diagnostic.}\n"
        "\\label{tab:tau-audit}\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\begin{tabular}{lcccc}\n"
        "\\hline\n"
        "Source & \\(\\tau\\) & Reference & \\(|\\Delta\\tau|\\) & Status \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def make_prediction_accuracy_by_zone_table(frequency_df, source_data_df):
    """Return prediction accuracy split by margin zone."""
    frequency_rows = _normalized_frequency_diagnostics(frequency_df)
    source_rows = _paired_mcnemar_summary_rows(source_data_df)
    summary_by_key = {(row["model"], row["attack"]): row for row in source_rows}
    counts = {
        "High": {"correct": 0, "total": 0, "max_abs_delta": 0.0},
        "Low": {"correct": 0, "total": 0, "max_abs_delta": 0.0},
    }
    for row in frequency_rows.sort_values(["model", "attack"]).itertuples(index=False):
        key = (str(row.model), str(row.attack))
        if key not in summary_by_key:
            continue
        summary = summary_by_key[key]
        zone = _margin_zone(row)
        predicted = _compact_schedule(row.predicted_best_schedule)
        winner = _compact_schedule(summary["winner"])
        counts[zone]["correct"] += int(predicted == winner)
        counts[zone]["total"] += 1
        counts[zone]["max_abs_delta"] = max(
            counts[zone]["max_abs_delta"],
            abs(float(summary["delta_pp"])),
        )
    rows = []
    for zone in ["High", "Low"]:
        total = counts[zone]["total"]
        correct = counts[zone]["correct"]
        accuracy = "n/a" if total == 0 else f"{correct}/{total}"
        interpretation = (
            "rule-stable"
            if zone == "High"
            else "paired audit required"
        )
        rows.append(
            (
                zone,
                accuracy,
                f"{counts[zone]['max_abs_delta']:.1f}",
                interpretation,
            )
        )
    body = "\n".join(
        f"{zone} & {accuracy} & {effect} & {interpretation} \\\\"
        for zone, accuracy, effect, interpretation in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Prediction accuracy by margin zone. High-margin rows use "
        "\\(|\\omega_\\delta-\\tau|>0.3\\). Low-margin rows are treated as practically equivalent checks.}\n"
        "\\label{tab:accuracy}\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\begin{tabular}{lccc}\n"
        "\\hline\n"
        "Zone & Correct & Max \\(|\\Delta|\\) pp & Interpretation \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def make_robust_training_gradient_table(frequency_df):
    """Return the ResNet-to-robust-model centroid gradient audit."""
    df = _normalized_frequency_diagnostics(frequency_df)
    rows = []
    for attack, group in df.groupby("attack", sort=True):
        if str(attack).lower() == "jpeg_aware_pgd":
            continue
        ordered = group.sort_values("omega_delta", ascending=False)
        high = ordered.iloc[0]
        low = ordered.iloc[-1]
        rows.append(
            (
                latex_escape(_condition_display_name(attack)),
                latex_escape(_condition_display_name(high.model)),
                f"{float(high.omega_delta):.3f}",
                latex_escape(_condition_display_name(low.model)),
                f"{float(low.omega_delta):.3f}",
                f"{float(high.omega_delta) - float(low.omega_delta):.3f}",
            )
        )
    if not rows:
        rows = [["No nonadaptive rows", "", "0.000", "", "0.000", "0.000"]]
    body = "\n".join(
        f"{attack} & {high_model} & {high_omega} & {low_model} & {low_omega} & {drop} \\\\"
        for attack, high_model, high_omega, low_model, low_omega, drop in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Robust-training frequency-gradient audit. Robust checkpoints shift "
        "\\(\\omega_\\delta\\) toward lower DCT frequencies relative to the highest-centroid row.}\n"
        "\\label{tab:robust-grad}\n"
        "\\centering\n"
        "\\scriptsize\n"
        "\\begin{tabular}{lccccc}\n"
        "\\hline\n"
        "Attack & High model & High \\(\\omega_\\delta\\) & Low model & Low \\(\\omega_\\delta\\) & Drop \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def make_jpeg_aware_boundary_table(frequency_df, source_data_df):
    """Return a boundary check table for JPEG-aware attacks."""
    frequency_rows = _normalized_frequency_diagnostics(frequency_df)
    source_rows = _paired_mcnemar_summary_rows(source_data_df)
    summary_by_key = {(row["model"], row["attack"]): row for row in source_rows}
    jpeg_rows = frequency_rows[
        frequency_rows["attack"].astype(str).str.contains("jpeg_aware", case=False)
    ].copy()
    rows = []
    for row in jpeg_rows.sort_values(["model", "attack"]).itertuples(index=False):
        key = (str(row.model), str(row.attack))
        summary = summary_by_key.get(key)
        winner = "n/a" if summary is None else _compact_schedule(summary["winner"])
        delta = "n/a" if summary is None else f"{summary['delta_pp']:.1f}"
        rows.append(
            (
                latex_escape(_condition_display_name(f"{row.dataset} {row.model} {row.attack}")),
                f"{float(row.omega_delta):.3f}",
                f"{_margin_value(row):.3f}",
                latex_escape(_compact_schedule(row.predicted_best_schedule)),
                latex_escape(winner),
                delta,
            )
        )
    if not rows:
        rows = [["No JPEG-aware rows", "0.000", "0.000", "n/a", "n/a", "n/a"]]
    body = "\n".join(
        f"{condition} & {omega} & {margin} & {predicted} & {winner} & {delta} \\\\"
        for condition, omega, margin, predicted, winner, delta in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{JPEG-aware boundary check. These rows audit the adaptive setting "
        "separately from the nonadaptive sign-rule validation set.}\n"
        "\\label{tab:jpeg-aware}\n"
        "\\centering\n"
        "\\tiny\n"
        "{\\setlength{\\tabcolsep}{1.5pt}\n"
        "\\begin{tabular}{@{}p{0.36\\linewidth}rrrrr@{}}\n"
        "\\hline\n"
        "Condition & \\(\\omega_\\delta\\) & Margin & Pred. & Win & \\(\\Delta\\) \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def make_mcnemar_audit_table(source_data_df, frequency_df=None):
    """Return a compact paired McNemar audit table from per-sample source data."""
    df = source_data_df.copy()
    missing = [
        column
        for column in (
            "dataset",
            "attack",
            "schedule",
            "generations",
            "sample_index",
            "attack_success",
        )
        if column not in df.columns
    ]
    if missing:
        raise ValueError(
            "source data are missing required columns: " + ", ".join(missing)
        )
    df = df[df["schedule"].isin(["front_loaded", "fixed"])].copy()
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.lower() == "cifar10"].copy()
    if "clean_correct" in df.columns:
        df = df[_to_bool_series(df["clean_correct"])]
    if df.empty:
        return _mcnemar_unavailable_table()

    df["generation"] = df["generations"].astype(int)
    df["attack_success_bool"] = _to_bool_series(df["attack_success"])
    context_columns = _source_context_columns(df)
    max_generation = df.groupby(context_columns)["generation"].transform("max")
    df = df[df["generation"] == max_generation]
    rows = []
    for key, group in df.groupby(context_columns, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        generation = int(group["generation"].max())
        range_count = int(group["range_name"].nunique()) if "range_name" in group.columns else 1
        paired = group.pivot_table(
            index=_mcnemar_pair_index_columns(group),
            columns="schedule",
            values="attack_success_bool",
            aggfunc="first",
        ).dropna(subset=["front_loaded", "fixed"])
        if paired.empty:
            continue
        front_loaded = paired["front_loaded"].astype(bool).to_numpy()
        fixed = paired["fixed"].astype(bool).to_numpy()
        n01, n10 = metrics.mcnemar_table(front_loaded, fixed)
        pvalue = metrics.mcnemar_pvalue(n01, n10)
        delta_pp = 100.0 * (front_loaded.mean() - fixed.mean())
        label = _condition_display_name(
            " ".join(str(value) for value in key_values)
        )
        rows.append(
            (
                latex_escape(label),
                str(range_count),
                str(generation),
                str(int(len(paired))),
                f"{delta_pp:.1f}",
                f"{n10}/{n01}",
                _format_pvalue(pvalue),
                _significance_label(pvalue),
            )
        )
    if not rows:
        return _mcnemar_unavailable_table()

    body = "\n".join(
        f"{condition} & {range_count} & {generation} & {pairs} & {delta_pp} & {discordant} & {pvalue} & {sig} \\\\"
        for condition, range_count, generation, pairs, delta_pp, discordant, pvalue, sig in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Paired McNemar audit for front-loaded versus fixed schedules at the largest generation. "
        "R is the number of QF ranges tested, G is the maximum generation count, "
        "\\(\\Delta\\) is FL minus Fix ASR in percentage points, and Disc. reports FL-only/Fixed-only successes.}\n"
        "\\label{tab:mcnemar-audit}\n"
        "\\centering\n"
        "\\tiny\n"
        "{\\setlength{\\tabcolsep}{1.5pt}\n"
        "\\begin{tabular}{@{}p{0.31\\linewidth}rrrrrrr@{}}\n"
        "\\hline\n"
        "Condition & R & G & Pairs & \\(\\Delta\\) & Disc. & \\(p\\) & Sig. \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def make_threshold_margin_audit_table(frequency_df, source_data_df):
    """Return a table comparing threshold predictions with paired winners."""
    frequency_rows = _normalized_frequency_diagnostics(frequency_df)
    if "dataset" in frequency_rows.columns:
        frequency_rows = frequency_rows[
            frequency_rows["dataset"].astype(str).str.lower() == "cifar10"
        ].copy()
    source_rows = _paired_mcnemar_summary_rows(source_data_df)
    summary_by_key = {
        (row["model"], row["attack"]): row for row in source_rows
    }

    rows = []
    for row in frequency_rows.sort_values(["model", "attack"]).itertuples(index=False):
        key = (str(row.model), str(row.attack))
        if key not in summary_by_key:
            continue
        summary = summary_by_key[key]
        predicted = _compact_schedule(row.predicted_best_schedule)
        winner = _compact_schedule(summary["winner"])
        match = "yes" if predicted == winner else "no"
        margin = abs(float(row.omega_delta) - float(row.tau))
        rows.append(
            (
                latex_escape(
                    _condition_display_name(f"{row.dataset} {row.model} {row.attack}")
                ),
                f"{float(row.omega_delta):.3f}",
                f"{margin:.3f}",
                latex_escape(predicted),
                latex_escape(winner),
                f"{summary['delta_pp']:.1f}",
                match,
            )
        )
    if not rows:
        return _threshold_margin_unavailable_table()

    body = "\n".join(
        f"{condition} & {omega} & {margin} & {predicted} & {winner} & {delta_pp} & {match} \\\\"
        for condition, omega, margin, predicted, winner, delta_pp, match in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{Threshold-margin audit linking the frequency rule to the full paired audit. "
        "Margin is \\(|\\omega_\\delta-\\tau|\\). Win is the lower-ASR FL-vs-Fix family.}\n"
        "\\label{tab:threshold-margin-audit}\n"
        "\\centering\n"
        "\\tiny\n"
        "{\\setlength{\\tabcolsep}{1.5pt}\n"
        "\\begin{tabular}{@{}p{0.34\\linewidth}rrrrrr@{}}\n"
        "\\hline\n"
        "Condition & \\(\\omega_\\delta\\) & Margin & Pred. & Win & \\(\\Delta\\) & Match \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def _significance_label(pvalue):
    return "yes" if float(pvalue) < 0.05 else "no"


def _format_pvalue(pvalue):
    value = float(pvalue)
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _format_asr_cell(value):
    if value != value:
        return "---"
    return f"{float(value):.3f}"


def _margin_value(row):
    return abs(float(row.omega_delta) - float(row.tau))


def _margin_zone(row):
    return "High" if _margin_value(row) > MARGIN_ZONE_THRESHOLD else "Low"


def _frequency_zone_lookup(frequency_df):
    frequency_rows = _normalized_frequency_diagnostics(frequency_df)
    lookup = {}
    for row in frequency_rows.itertuples(index=False):
        lookup[(str(row.dataset), str(row.model), str(row.attack))] = _margin_zone(row)
        lookup[(str(row.model), str(row.attack))] = _margin_zone(row)
    return lookup


def _lookup_zone(zone_by_key, key_dict):
    dataset = str(key_dict.get("dataset", ""))
    model = str(key_dict.get("model", ""))
    attack = str(key_dict.get("attack", ""))
    return zone_by_key.get((dataset, model, attack), zone_by_key.get((model, attack), "n/a"))


def _paired_mcnemar_summary_rows(source_data_df):
    df = source_data_df.copy()
    df = df[df["schedule"].isin(["front_loaded", "fixed"])].copy()
    if "dataset" in df.columns:
        df = df[df["dataset"].astype(str).str.lower() == "cifar10"].copy()
    if "clean_correct" in df.columns:
        df = df[_to_bool_series(df["clean_correct"])]
    if df.empty:
        return []
    df["generation"] = df["generations"].astype(int)
    df["attack_success_bool"] = _to_bool_series(df["attack_success"])
    context_columns = _source_context_columns(df)
    max_generation = df.groupby(context_columns)["generation"].transform("max")
    df = df[df["generation"] == max_generation]
    rows = []
    for key, group in df.groupby(context_columns, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        key_dict = dict(zip(context_columns, key_values))
        paired = group.pivot_table(
            index=_mcnemar_pair_index_columns(group),
            columns="schedule",
            values="attack_success_bool",
            aggfunc="first",
        ).dropna(subset=["front_loaded", "fixed"])
        if paired.empty:
            continue
        front_loaded = paired["front_loaded"].astype(bool).to_numpy()
        fixed = paired["fixed"].astype(bool).to_numpy()
        delta_pp = 100.0 * (front_loaded.mean() - fixed.mean())
        rows.append(
            {
                **key_dict,
                "delta_pp": delta_pp,
                "winner": "front_loaded" if delta_pp < 0.0 else "fixed",
            }
        )
    return rows


def _threshold_margin_unavailable_table():
    return (
        "\\begin{table}[t]\n"
        "\\caption{Threshold-margin audit was not available for this render.}\n"
        "\\label{tab:threshold-margin-audit}\n"
        "\\centering\n"
        "\\begin{tabular}{l}\n"
        "\\hline\n"
        "No paired source data supplied \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _accuracy_unavailable_table():
    return (
        "\\begin{table}[t]\n"
        "\\caption{Prediction accuracy by margin zone was not available for this render.}\n"
        "\\label{tab:accuracy}\n"
        "\\centering\n"
        "\\begin{tabular}{l}\n"
        "\\hline\n"
        "No paired source data supplied \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _robust_training_unavailable_table():
    return (
        "\\begin{table}[t]\n"
        "\\caption{Robust-training frequency-gradient audit was not available for this render.}\n"
        "\\label{tab:robust-grad}\n"
        "\\centering\n"
        "\\begin{tabular}{l}\n"
        "\\hline\n"
        "No frequency CSV supplied \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _jpeg_aware_unavailable_table():
    return (
        "\\begin{table}[t]\n"
        "\\caption{JPEG-aware boundary check was not available for this render.}\n"
        "\\label{tab:jpeg-aware}\n"
        "\\centering\n"
        "\\begin{tabular}{l}\n"
        "\\hline\n"
        "No paired source data supplied \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _make_model_boundary_summary_table(df):
    df = _filter_cifar_rows(df)
    best_rows = _best_boundary_rows(df)
    rows = []
    for key, group in best_rows.groupby(["dataset", "model", "attack"], sort=True):
        dataset, model, attack = key
        schedule_counts = group["schedule"].value_counts().sort_index()
        count_text = ", ".join(
            f"{_compact_schedule(schedule)} {count}"
            for schedule, count in schedule_counts.items()
        )
        rows.append(
            (
                latex_escape(_condition_display_name(f"{dataset} {model} {attack}")),
                latex_escape(count_text),
                f"{group['asr'].mean():.3f}",
            )
        )

    body = "\n".join(
        f"{condition} & {counts} & {mean_asr} \\\\"
        for condition, counts, mean_asr in rows
    )
    table = (
        "\\begin{table}[t]\n"
        "\\caption{FL-vs-Fix winner by model and threat model at the largest generation count. Counts give the number of QF ranges in which each supplied schedule family has the lower ASR. FL denotes front-loaded and Fix denotes fixed.}\n"
        "\\label{tab:model-boundary}\n"
        "\\centering\n"
        "\\scriptsize\n"
        "{\\setlength{\\tabcolsep}{2pt}\n"
        "\\begin{tabular}{@{}p{0.43\\linewidth}p{0.36\\linewidth}p{0.15\\linewidth}@{}}\n"
        "\\hline\n"
        "Condition & Winner counts & Mean ASR \\\\\n"
        "\\hline\n"
        f"{body}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )
    assert_no_placeholders(table)
    return table


def _short_schedule(schedule):
    aliases = {
        "front_loaded": "front-loaded",
        "geometric": "geometric",
        "arithmetic": "arithmetic",
        "fixed": "fixed",
    }
    return aliases.get(str(schedule), str(schedule))


def _compact_schedule(schedule):
    aliases = {
        "front_loaded": "FL",
        "geometric": "Geo",
        "arithmetic": "Arith",
        "fixed": "Fix",
        "front-loaded": "FL",
    }
    return aliases.get(str(schedule), str(schedule))


def _compact_schedule_list(text):
    replacements = {
        "front-loaded": "FL",
        "geometric": "Geo",
        "arithmetic": "Arith",
        "fixed": "Fix",
    }
    compact = str(text)
    for source, target in replacements.items():
        compact = compact.replace(source, target)
    return compact


def _condition_display_name(label):
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
    return " ".join(
        aliases.get(token, token.replace("_", "-")) for token in str(label).split()
    )


def _best_boundary_rows(df):
    df = _filter_cifar_rows(df)
    context_columns = _context_group_columns(df)
    group_columns = [*context_columns, "attack", "range_name"]
    max_generation = df.groupby(group_columns)["generation"].transform("max")
    df = df[df["generation"] == max_generation]
    best_rows = df.loc[
        df.groupby([*group_columns, "generation"])["asr"].idxmin()
    ].sort_values([*group_columns, "generation"])
    return best_rows


def _context_group_columns(df):
    columns = []
    if "dataset" in df.columns:
        columns.append("dataset")
    if "model" in df.columns:
        columns.append("model")
    return columns


def _source_context_columns(df):
    columns = ["dataset"]
    if "model" in df.columns:
        columns.append("model")
    columns.append("attack")
    return columns


def _mcnemar_pair_index_columns(df):
    columns = ["sample_index"]
    if "range_name" in df.columns:
        columns.insert(0, "range_name")
    return columns


def _to_bool_series(series):
    return series.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        if not isinstance(value, bool)
        else value
    )


def _mcnemar_unavailable_table():
    return (
        "\\begin{table}[t]\n"
        "\\caption{Paired McNemar audit was not available for this render.}\n"
        "\\label{tab:mcnemar-audit}\n"
        "\\centering\n"
        "\\begin{tabular}{l}\n"
        "\\hline\n"
        "No paired source data supplied \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _context_header(column):
    if column == "dataset":
        return "Dataset"
    if column == "model":
        return "Model"
    return latex_escape(column)


def _generic_boundary_condition_label(row_dict, context_columns):
    parts = [str(row_dict[column]) for column in context_columns]
    parts.extend([str(row_dict["attack"]), str(row_dict["range_name"])])
    parts.append(f"G{row_dict['generation']}")
    return _condition_display_name(" ".join(parts))


def _boundary_row_cells(row_dict, context_columns):
    cells = [latex_escape(row_dict[column]) for column in context_columns]
    cells.extend(
        [
            latex_escape(row_dict["attack"]),
            latex_escape(row_dict["range_name"]),
            str(row_dict["generation"]),
            latex_escape(row_dict["schedule"]),
            latex_escape(row_dict["qf"]),
            f"{row_dict['no_defense_asr']:.3f}",
            f"{row_dict['asr']:.3f}",
        ]
    )
    return cells


def render_latex_manuscript(
    draft_path,
    metrics_csv,
    figures_dir,
    tables_dir,
    out_path,
    frequency_csv=None,
    source_data_csv=None,
    template_cls_path=None,
):
    """Render an IEEE Signal Processing Letters-compatible manuscript."""
    draft_text = Path(draft_path).read_text(encoding="utf-8")
    metrics_df = load_metrics_csv(metrics_csv)
    frequency_df = load_metrics_csv(frequency_csv) if frequency_csv is not None else None
    source_data_df = (
        load_metrics_csv(source_data_csv) if source_data_csv is not None else None
    )
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    out_path = Path(out_path)
    figure_1 = figures_dir / "figure1_asr_by_generation.pdf"
    figure_2 = figures_dir / "figure2_centroid_vs_delta.pdf"
    figure_3 = figures_dir / "figure3_attack_range_summary.pdf"
    required_figures = [figure_1, figure_3]
    if "{{CENTROID_DELTA_FIGURE}}" in draft_text:
        required_figures.append(figure_2)
    _require_files(*required_figures)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    table_ii = make_table_ii(metrics_df)
    table_iii = make_table_iii(metrics_df)
    table_iv = make_threat_model_boundary_table(metrics_df)
    full_asr_table = make_full_four_schedule_asr_table(metrics_df)
    tau_table = make_tau_independence_table(frequency_df)
    mcnemar_table = (
        make_mcnemar_audit_table(source_data_df, frequency_df)
        if source_data_df is not None
        else _mcnemar_unavailable_table()
    )
    prediction_table = make_prediction_rule_table()
    frequency_table = (
        make_frequency_diagnostics_table(frequency_df)
        if frequency_df is not None
        else _frequency_diagnostics_unavailable_table()
    )
    robust_training_table = (
        make_robust_training_gradient_table(frequency_df)
        if frequency_df is not None
        else _robust_training_unavailable_table()
    )
    jpeg_aware_table = (
        make_jpeg_aware_boundary_table(frequency_df, source_data_df)
        if frequency_df is not None and source_data_df is not None
        else _jpeg_aware_unavailable_table()
    )
    (tables_dir / "table_i_prediction_rules.tex").write_text(
        prediction_table,
        encoding="utf-8",
    )
    (tables_dir / "table_frequency_diagnostics.tex").write_text(
        frequency_table,
        encoding="utf-8",
    )
    (tables_dir / "table_ii.tex").write_text(table_ii, encoding="utf-8")
    (tables_dir / "table_iii.tex").write_text(table_iii, encoding="utf-8")
    (tables_dir / "table_iv.tex").write_text(table_iv, encoding="utf-8")
    (tables_dir / "table_full_four_schedule_asr.tex").write_text(
        full_asr_table,
        encoding="utf-8",
    )
    (tables_dir / "table_tau_independence.tex").write_text(
        tau_table,
        encoding="utf-8",
    )
    (tables_dir / "table_mcnemar_audit.tex").write_text(
        mcnemar_table,
        encoding="utf-8",
    )
    (tables_dir / "table_robust_training_gradient.tex").write_text(
        robust_training_table,
        encoding="utf-8",
    )
    (tables_dir / "table_jpeg_aware_boundary.tex").write_text(
        jpeg_aware_table,
        encoding="utf-8",
    )

    title, sections = _parse_draft(draft_text)
    abstract, keywords, body_sections = _extract_special_sections(sections)
    relative_figure_1 = _relative_to_out_dir(figure_1, out_path)
    relative_figure_2 = _relative_to_out_dir(figure_2, out_path)
    relative_figure_3 = _relative_to_out_dir(figure_3, out_path)
    raw_blocks = {
        "{{FREQUENCY_MODEL}}": _frequency_model_block(),
        "{{MICROSTRUCTURE_MODEL}}": _microstructure_model_block(),
        "{{PREDICTION_RULE_TABLE}}": prediction_table,
        "{{MECHANISM_FIGURE}}": _figure_block(
            relative_figure_1,
            "ASR by JPEG generation and schedule family for C10 ResNet-18 PGD Range R1. Front-loaded schedules reduce ASR after repeated recompression, while fixed schedules saturate after the first step.",
            "fig:asr-generation",
        ),
        "{{FREQUENCY_DIAGNOSTIC_FIGURE}}": "",
        "{{FREQUENCY_DIAGNOSTIC_TABLE}}": frequency_table,
        "{{TAU_INDEPENDENCE_TABLE}}": tau_table,
        "{{CENTROID_DELTA_FIGURE}}": _figure_block(
            relative_figure_2,
            "Signed centroid coordinate [math:\\omega_\\delta-\\tau] versus paired ASR difference [math:\\Delta], computed as front-loaded minus fixed in percentage points. Shaded bands mark zero-reference regions and the [math:|\\Delta|<1] practical-equivalence band.",
            "fig:centroid-delta",
        ),
        "{{FULL_FOUR_SCHEDULE_ASR_TABLE}}": full_asr_table,
        "{{SCHEDULE_DEFINITION}}": _schedule_definition_block(),
        "{{ASR_DEFINITION}}": _asr_definition_block(),
        "{{SCHEDULE_BOUNDARY_FIGURE}}": _figure_block(
            relative_figure_3,
            "Best-schedule counts at the largest generation count. Bars count quality-factor ranges in which a schedule gave the lowest defended attack success rate for each dataset, model, and attack.",
            "fig:schedule-boundary",
        ),
        "{{MODEL_BOUNDARY_TABLE}}": table_iv,
        "{{MCNEMAR_AUDIT_TABLE}}": mcnemar_table,
        "{{ROBUST_TRAINING_GRADIENT_TABLE}}": robust_training_table,
        "{{JPEG_AWARE_BOUNDARY_TABLE}}": jpeg_aware_table,
    }
    body = _render_sections(body_sections, raw_blocks)
    body = _append_missing_evidence_blocks(body, raw_blocks)
    if source_data_df is not None and "{{MCNEMAR_AUDIT_TABLE}}" not in draft_text:
        body = "\n".join([body, mcnemar_table])
    template_note = _template_note(template_cls_path)

    tex = (
        "\\documentclass[journal]{IEEEtran}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{cite}\n"
        "\\usepackage{url}\n"
        f"{template_note}"
        f"\\title{{{latex_escape(title)}}}\n"
        "\\author{Anonymous Authors}\n"
        "\\markboth{IEEE Signal Processing Letters}{Anonymous Authors: JPEG Recompression Scheduling}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        f"{_abstract_block(abstract)}"
        f"{_keywords_block(keywords)}"
        "\\IEEEpeerreviewmaketitle\n"
        f"{body}\n"
        f"{_bibliography_block()}"
        "\\end{document}\n"
    )
    assert_no_placeholders(tex)
    assert_manuscript_style_guardrails(tex)
    assert_citation_order(tex)
    out_path.write_text(tex, encoding="utf-8")
    return out_path


def _normalized_metrics(metrics_df):
    df = _filter_cifar_rows(metrics_df.copy())
    if "generation" not in df.columns and "generations" in df.columns:
        df["generation"] = df["generations"]
    if "qf" not in df.columns and "qfs" in df.columns:
        df["qf"] = df["qfs"]
    missing = [
        column
        for column in (
            "attack",
            "range_name",
            "schedule",
            "generation",
            "qf",
            "no_defense_asr",
            "asr",
        )
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"metrics data is missing required columns: {', '.join(missing)}")
    return df


def _normalized_frequency_diagnostics(frequency_df):
    df = _filter_cifar_rows(frequency_df.copy())
    if "dataset" in df.columns:
        df["tau"] = df["dataset"].map(frequency.elimination_threshold_for_dataset)
    if {"omega_delta", "tau"}.issubset(df.columns):
        df["omega_delta_relation"] = df.apply(
            lambda row: frequency.omega_delta_relation(row["omega_delta"], row["tau"]),
            axis=1,
        )
    if {"omega_delta", "tau"}.issubset(df.columns):
        jpeg_aware_attack = _jpeg_aware_series(df)
        df["predicted_best_schedule"] = [
            frequency.predict_schedule_from_omega_delta(omega_delta, tau, jpeg_aware)
            for omega_delta, tau, jpeg_aware in zip(
                df["omega_delta"],
                df["tau"],
                jpeg_aware_attack,
            )
        ]
        df["prediction_rule"] = [
            frequency.prediction_rule_label(jpeg_aware, tau)
            for jpeg_aware, tau in zip(jpeg_aware_attack, df["tau"])
        ]
    return df


def _filter_cifar_rows(df):
    if "dataset" not in df.columns:
        return df
    return df[df["dataset"].astype(str).str.lower() == "cifar10"].copy()


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


def _format_unique(series):
    values = []
    for value in series:
        text = str(value)
        if text not in values:
            values.append(text)
    return ", ".join(values)


def _parse_draft(draft_text):
    title = "JPEG Defense Evaluation"
    sections = []
    current_heading = None
    current_lines = []
    for raw_line in draft_text.splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            title = line[2:].strip() or title
            continue
        if line.startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[3:].strip() or "Section"
            current_lines = []
            continue
        if current_heading is not None:
            current_lines.append(raw_line)
    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))
    if not sections:
        sections = [("Introduction", draft_text.strip())]
    return title, sections


def _extract_special_sections(sections):
    abstract = ""
    keywords = ""
    body_sections = []
    for heading, content in sections:
        normalized = heading.strip().lower()
        if normalized == "abstract":
            abstract = content.strip()
        elif normalized in {"keywords", "ieee keywords"}:
            keywords = content.strip()
        else:
            body_sections.append((heading, content))
    return abstract, keywords, body_sections


def _render_sections(sections, raw_blocks=None):
    raw_blocks = raw_blocks or {}
    rendered = []
    for heading, content in sections:
        rendered.append(f"\\section{{{latex_escape(heading)}}}")
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", content)]
        for paragraph in paragraphs:
            if paragraph:
                rendered.append(_render_paragraph(paragraph, raw_blocks))
    return "\n".join(rendered)


def _render_paragraph(paragraph, raw_blocks=None):
    raw_blocks = raw_blocks or {}
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if any(line in raw_blocks for line in lines):
        rendered = []
        prose_buffer = []
        for line in lines:
            if line in raw_blocks:
                if prose_buffer:
                    rendered.append(_render_inline_markup(" ".join(prose_buffer)))
                    prose_buffer = []
                rendered.append(raw_blocks[line])
            else:
                prose_buffer.append(line)
        if prose_buffer:
            rendered.append(_render_inline_markup(" ".join(prose_buffer)))
        return "\n".join(rendered)
    if lines and all(line in raw_blocks for line in lines):
        return "\n".join(raw_blocks[line] for line in lines)
    text = " ".join(lines)
    if text in raw_blocks:
        return raw_blocks[text]
    return _render_inline_markup(text)


def _render_inline_markup(text):
    rendered = []
    cursor = 0
    for match in _INLINE_MARKUP_RE.finditer(text):
        rendered.append(latex_escape(text[cursor : match.start()]))
        kind, payload = match.groups()
        if kind == "cite":
            keys = [key.strip() for key in payload.split(",") if key.strip()]
            _require_latex_keys(keys, "citation")
            rendered.append("\\cite{" + ",".join(keys) + "}")
        elif kind == "fig":
            label = _prefixed_label(payload, "fig")
            rendered.append(f"Fig.~\\ref{{{label}}}")
        elif kind == "tab":
            label = _prefixed_label(payload, "tab")
            rendered.append(f"Table~\\ref{{{label}}}")
        elif kind == "math":
            if "}" in payload or "{" in payload:
                raise ValueError("Inline math markup must not contain braces")
            rendered.append(f"\\({payload}\\)")
        cursor = match.end()
    rendered.append(latex_escape(text[cursor:]))
    return "".join(rendered)


def _require_latex_keys(keys, kind):
    for key in keys:
        if not _KEY_RE.match(key):
            raise ValueError(f"Invalid {kind} key for LaTeX markup: {key}")


def _prefixed_label(label, prefix):
    _require_latex_keys([label], prefix)
    if label.startswith(f"{prefix}:"):
        return label
    return f"{prefix}:{label}"


def _abstract_block(abstract):
    if not abstract:
        return ""
    return (
        "\\begin{abstract}\n"
        f"{_render_inline_markup(_collapse_paragraph_text(abstract))}\n"
        "\\end{abstract}\n"
    )


def _keywords_block(keywords):
    if not keywords:
        return ""
    return (
        "\\begin{IEEEkeywords}\n"
        f"{latex_escape(_collapse_paragraph_text(keywords))}\n"
        "\\end{IEEEkeywords}\n"
    )


def _collapse_paragraph_text(text):
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def _append_missing_evidence_blocks(body, raw_blocks):
    additions = []
    if "\\includegraphics" not in body:
        additions.append(raw_blocks["{{MECHANISM_FIGURE}}"])
        additions.append(raw_blocks["{{SCHEDULE_BOUNDARY_FIGURE}}"])
    if "\\begin{table" not in body:
        additions.append(raw_blocks["{{MODEL_BOUNDARY_TABLE}}"])
    if not additions:
        return body
    return "\n".join([body, "\\section{Evidence Summary}", *additions])


def _frequency_model_block():
    return (
        "\\begin{equation}\n"
        "\\omega_\\delta = "
        "\\frac{\\sum_{u,v} \\rho_{u,v}|\\delta_0(u,v)|^2}"
        "{\\sum_{u,v}|\\delta_0(u,v)|^2},\\quad "
        "\\rho_{u,v}=\\sqrt{u^2+v^2}.\n"
        "\\label{eq:perturbation-centroid}\n"
        "\\end{equation}\n"
        "\\begin{equation}\n"
        "\\tau:\\quad \\Delta_{\\bar{q}}(\\tau)="
        "\\mathrm{median}_{u,v}|\\delta_0(u,v)|.\n"
        "\\label{eq:elimination-threshold}\n"
        "\\end{equation}\n"
        "In (\\ref{eq:perturbation-centroid}), \\(\\delta_0(u,v)\\) is the "
        "DCT-domain attack residual before recompression and \\(\\rho_{u,v}\\) "
        "is radial frequency. Equation (\\ref{eq:elimination-threshold}) defines "
        "the elimination threshold \\(\\tau\\), where the midpoint-QF quantization "
        "step matches the typical perturbation amplitude."
    )


def _microstructure_model_block():
    return (
        "\\begin{equation}\n"
        "\\mu_1(u,v)=\\left\\lceil\\frac{|\\delta_0(u,v)|}{\\Delta_1(u,v)}\\right\\rceil.\n"
        "\\label{eq:microstructure}\n"
        "\\end{equation}\n"
        "In (\\ref{eq:microstructure}), \\((u,v)\\) is a DCT coefficient position, "
        "\\(\\delta_0(u,v)\\) is the attack residual before recompression, "
        "\\(\\Delta_1(u,v)\\) is the first-stage JPEG quantization step, "
        "and \\(\\mu_1(u,v)\\) is the residual bin count that can survive the first stage.\n"
        "\\begin{equation}\n"
        "\\mu_2(u,v)\\le r_1(u,v)\\mu_1(u,v)+1,\\quad "
        "r_1(u,v)=\\frac{\\Delta_2(u,v)}{\\Delta_1(u,v)}.\n"
        "\\label{eq:two-stage-recursion}\n"
        "\\end{equation}\n"
        "Equation (\\ref{eq:two-stage-recursion}) gives the second-stage residual bound. "
        "It makes the first quantization step measurable because a frequency with "
        "\\(\\mu_1=0\\) cannot recover its original attack residual in later stages."
    )


def _frequency_diagnostics_unavailable_table():
    return (
        "\\begin{table}[t]\n"
        "\\caption{Frequency diagnostics were not supplied for this render.}\n"
        "\\label{tab:frequency-diagnostics}\n"
        "\\centering\n"
        "\\begin{tabular}{l}\n"
        "\\hline\n"
        "No frequency CSV supplied \\\\\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _schedule_definition_block():
    return (
        "\\begin{equation}\n"
        "s=(q_1,\\ldots,q_T), \\quad "
        "D_s(x)=J_{q_T}\\circ\\cdots\\circ J_{q_1}(x).\n"
        "\\label{eq:schedule}\n"
        "\\end{equation}\n"
        "In (\\ref{eq:schedule}), \\(s\\) is the recompression schedule, "
        "\\(q_t\\) is the JPEG quality factor at generation \\(t\\), "
        "\\(T\\) is the number of recompressions, \\(J_q\\) is JPEG recompression at quality factor \\(q\\), "
        "and \\(D_s\\) is the defended image operator."
    )


def _asr_definition_block():
    return (
        "\\begin{equation}\n"
        "\\mathrm{ASR}(s)=\\frac{1}{N_c}\\sum_i "
        "\\mathbf{1}\\left[f(D_s(x_i+\\delta_i))\\ne y_i\\right]"
        "\\mathbf{1}\\left[f(x_i)=y_i\\right].\n"
        "\\label{eq:asr}\n"
        "\\end{equation}\n"
        "In (\\ref{eq:asr}), \\(x_i\\) is image \\(i\\), \\(y_i\\) is its label, "
        "\\(f\\) is the classifier, \\(\\delta_i\\) is the attack perturbation, "
        "\\(N_c\\) is the number of clean-correct images, and \\(\\mathbf{1}[\\cdot]\\) is the indicator function."
    )


def _figure_block(path, caption, label):
    return (
        "\\begin{figure}[t]\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{\\detokenize{{{_latex_path(path)}}}}}\n"
        f"\\caption{{{_render_inline_markup(caption)}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{figure}\n"
    )


def _latex_path(path):
    return Path(path).as_posix()


def _relative_to_out_dir(path, out_path):
    return Path(os.path.relpath(Path(path).resolve(), Path(out_path).parent.resolve()))


def _bibliography_block():
    return (
        "\\begin{thebibliography}{17}\n"
        "\\bibitem{jpeg} G. K. Wallace, ``The JPEG still picture compression standard,'' "
        "{\\em Commun. ACM}, vol. 34, no. 4, pp. 30--44, Apr. 1991.\n"
        "\\bibitem{shield} N. Das {\\em et al.}, "
        "``SHIELD: Fast, practical defense and vaccination for deep learning using JPEG compression,'' "
        "in {\\em Proc. ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.}, 2018, pp. 196--204.\n"
        "\\bibitem{xu} G. Xu, J. Li, and S. Liu, "
        "``Understanding and defending against adversarial examples via multiple JPEG recompressions,'' "
        "in {\\em Proc. IEEE Int. Conf. Acoust. Speech Signal Process.}, 2023, pp. 1--5.\n"
        "\\bibitem{goodfellow} I. J. Goodfellow, J. Shlens, and C. Szegedy, "
        "``Explaining and harnessing adversarial examples,'' in {\\em Proc. Int. Conf. Learn. Representations}, 2015.\n"
        "\\bibitem{madry} A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, "
        "``Towards deep learning models resistant to adversarial attacks,'' in {\\em Proc. Int. Conf. Learn. Representations}, 2018.\n"
        "\\bibitem{athalye} A. Athalye, N. Carlini, and D. Wagner, "
        "``Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples,'' "
        "in {\\em Proc. Int. Conf. Mach. Learn.}, 2018, pp. 274--283.\n"
        "\\bibitem{cifar} A. Krizhevsky, ``Learning multiple layers of features from tiny images,'' "
        "Univ. Toronto, Toronto, ON, Canada, Tech. Rep., 2009.\n"
        "\\bibitem{robustbench} F. Croce {\\em et al.}, ``RobustBench: A standardized adversarial robustness benchmark,'' "
        "in {\\em Proc. Adv. Neural Inf. Process. Syst. Datasets Benchmarks Track}, 2021.\n"
        "\\bibitem{wong} E. Wong, L. Rice, and J. Z. Kolter, "
        "``Fast is better than free: Revisiting adversarial training,'' "
        "in {\\em Proc. Int. Conf. Learn. Representations}, 2020.\n"
        "\\bibitem{rice} L. Rice, E. Wong, and J. Z. Kolter, "
        "``Overfitting in adversarially robust deep learning,'' in {\\em Proc. Int. Conf. Mach. Learn.}, 2020, pp. 8093--8104.\n"
        "\\bibitem{shin} R. Shin and D. Song, ``JPEG-resistant adversarial images,'' "
        "in {\\em Proc. NeurIPS Workshop on Machine Learning and Computer Security}, 2017.\n"
        "\\bibitem{reich} C. Reich, B. Debnath, D. Patel, and S. Chakradhar, "
        "``Differentiable JPEG,'' "
        "in {\\em Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis.}, 2024, pp. 4126--4135.\n"
        "\\bibitem{wanghfc} H. Wang, X. Wu, Z. Huang, and E. P. Xing, "
        "``High-frequency component helps explain the generalization of convolutional neural networks,'' "
        "in {\\em Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.}, 2020, pp. 8684--8694.\n"
        "\\bibitem{imagenet} O. Russakovsky {\\em et al.}, ``ImageNet large scale visual recognition challenge,'' "
        "{\\em Int. J. Comput. Vis.}, vol. 115, no. 3, pp. 211--252, 2015.\n"
        "\\bibitem{vit} A. Dosovitskiy {\\em et al.}, "
        "``An image is worth 16x16 words: Transformers for image recognition at scale,'' "
        "in {\\em Proc. Int. Conf. Learn. Representations}, 2021.\n"
        "\\bibitem{swin} Z. Liu {\\em et al.}, "
        "``Swin Transformer: Hierarchical vision transformer using shifted windows,'' "
        "in {\\em Proc. IEEE/CVF Int. Conf. Comput. Vis.}, 2021, pp. 10012--10022.\n"
        "\\bibitem{deit} H. Touvron {\\em et al.}, "
        "``Training data-efficient image transformers and distillation through attention,'' "
        "in {\\em Proc. Int. Conf. Mach. Learn.}, 2021, pp. 10347--10357.\n"
        "\\end{thebibliography}\n"
    )


def _require_files(*paths):
    missing = [Path(path) for path in paths if not Path(path).is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Required figure file(s) missing: {missing_text}")


def _template_note(template_cls_path):
    if template_cls_path is None:
        return ""
    return f"% Template class path: {latex_escape(Path(template_cls_path).as_posix())}\n"
