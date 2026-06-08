"""Command-line entry point for offline experiment smoke runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jpeg_defense import experiments  # noqa: E402
from jpeg_defense.paths import CapsulePaths  # noqa: E402


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value!r}")
    return parsed


def _parse_ranges(value):
    ranges = []
    for item in str(value).split(","):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                f"ranges must use NAME:START:END entries, got {item!r}"
            )
        name, start_qf, end_qf = parts
        ranges.append((name, int(start_qf), int(end_qf)))
    if not ranges:
        raise argparse.ArgumentTypeError("ranges must contain at least one entry")
    return ranges


def _parse_generations(value):
    generations = tuple(int(item) for item in str(value).split(",") if item)
    if not generations:
        raise argparse.ArgumentTypeError("generations must contain at least one value")
    return generations


def _parse_models(value):
    models = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not models:
        raise argparse.ArgumentTypeError("models must contain at least one name")
    return models


def _parse_attacks(value):
    attacks = tuple(
        item.strip().replace("-", "_") for item in str(value).split(",") if item.strip()
    )
    if not attacks:
        raise argparse.ArgumentTypeError("attacks must contain at least one name")
    return attacks


def _parse_model_attacks(value):
    pairs = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":", maxsplit=1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"model-attacks entries must use MODEL:ATTACK, got {item!r}"
            )
        model_name, attack_name = parts
        model_name = model_name.strip()
        attack_name = attack_name.strip().replace("-", "_")
        if not model_name or not attack_name:
            raise argparse.ArgumentTypeError(
                f"model-attacks entries must use MODEL:ATTACK, got {item!r}"
            )
        pairs.append((model_name, attack_name))
    if not pairs:
        raise argparse.ArgumentTypeError("model-attacks must contain at least one pair")
    return tuple(pairs)


def main(argv=None):
    paths = CapsulePaths.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[
            "smoke",
            "cifar",
            "imagenet-smoke",
            "multimodel",
            "frequency",
            "full",
            "mcnemar-audit",
        ],
        default="smoke",
    )
    parser.add_argument("--data-root", type=Path, default=paths.data_root)
    parser.add_argument("--results-dir", type=Path, default=paths.results_root)
    parser.add_argument("--max-samples", type=_positive_int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--attack",
        choices=["fgsm", "pgd", "jpeg-aware-pgd"],
        default="fgsm",
    )
    parser.add_argument("--ranges", type=_parse_ranges, default=None)
    parser.add_argument("--generations", type=_parse_generations, default=(1, 2, 3))
    parser.add_argument("--models", type=_parse_models, default=None)
    parser.add_argument("--attacks", type=_parse_attacks, default=None)
    parser.add_argument("--model-attacks", type=_parse_model_attacks, default=None)
    parser.add_argument("--imagenet-root", type=Path, default=None)
    parser.add_argument("--attack-batch-size", type=_positive_int, default=64)
    parser.add_argument("--sample-offset", type=int, default=0)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        if args.mode == "imagenet-smoke":
            raise ValueError("imagenet-smoke mode is not implemented yet")
        if args.mode == "frequency":
            summary = experiments.run_frequency_diagnostics(
                data_root=args.data_root,
                results_dir=args.results_dir,
                max_samples=args.max_samples,
                attack=args.attack.replace("-", "_"),
                attacks=args.attacks,
                device=args.device,
                model_names=args.models,
                model_attack_pairs=args.model_attacks,
                imagenet_root=args.imagenet_root,
                attack_batch_size=args.attack_batch_size,
                sample_offset=args.sample_offset,
            )
        elif args.mode == "mcnemar-audit":
            model_names = args.models or (
                "cifar_resnet18",
                "robustbench_engstrom2019",
                "robustbench_wong2020fast",
                "robustbench_rice2020",
            )
            summary = experiments.run_mcnemar_audit_sweep(
                data_root=args.data_root,
                results_dir=args.results_dir,
                max_samples=args.max_samples,
                attacks=args.attacks or ("fgsm", "pgd"),
                device=args.device,
                model_names=model_names,
                ranges=args.ranges,
                generation=max(args.generations),
                imagenet_root=args.imagenet_root,
                attack_batch_size=args.attack_batch_size,
                sample_offset=args.sample_offset,
            )
        elif args.mode in {"multimodel", "full"}:
            model_names = args.models
            if args.mode == "full" and model_names is None:
                model_names = (
                    "cifar_resnet18",
                    "robustbench_engstrom2019",
                    "robustbench_wong2020fast",
                    "robustbench_rice2020",
                )
            summary = experiments.run_multimodel_sweep(
                data_root=args.data_root,
                results_dir=args.results_dir,
                max_samples=args.max_samples,
                attack=args.attack.replace("-", "_"),
                device=args.device,
                model_names=model_names,
                ranges=args.ranges,
                generations=args.generations,
                imagenet_root=args.imagenet_root,
                attack_batch_size=args.attack_batch_size,
            )
        else:
            summary = experiments.run_cifar_smoke(
                data_root=args.data_root,
                results_dir=args.results_dir,
                max_samples=args.max_samples,
                attack=args.attack.replace("-", "_"),
                device=args.device,
                ranges=args.ranges,
                generations=args.generations,
            )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        print("No downloads are attempted by this experiment runner.", file=sys.stderr)
        return 1

    if args.mode == "frequency":
        print(
            f"{args.mode} complete: "
            f"diagnostics written to {summary['frequency_metrics_csv']}"
        )
    else:
        print(
            f"{args.mode} complete: "
            f"{summary['rows']} rows written to {summary['metrics_csv']}"
        )
    print(f"summary: {summary['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
