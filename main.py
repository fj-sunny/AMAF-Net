"""
统一入口：选择模型并做五折训练。

在项目根目录运行::

    python main.py --list-models
    python main.py --experiment ours_proco --ct-root /path/to/ct_roi --clinical-xlsx /path/to/clinical.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from config import USE_3_CLASS, get_class_maps
from models.registry import list_models
from train.trainer import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Missing-aware multimodal fusion unified runner")
    parser.add_argument("--ct-root", type=str, default="", help="NC ROI CT 根目录")
    parser.add_argument("--clinical-xlsx", type=str, default="", help="临床 Excel 路径")
    parser.add_argument("--out-root", type=str, default="./output/amaf_net", help="输出根目录")
    parser.add_argument(
        "--three-class",
        action="store_true",
        default=USE_3_CLASS,
        help="使用 3 类 (CS/PA/PC)；默认 4 类",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="",
        help=(
            "实验名称。例如 image_resnet18, fusion_concat, fusion_daft, "
            "ours, ours_proco, ours_A1, ours_proco_A1"
        ),
    )
    parser.add_argument("--list-models", action="store_true", help="列出可选模型后退出")
    parser.add_argument("--lambda-proco", type=float, default=0.1, help="ProCo 损失权重，仅 *_proco 实验使用")
    parser.add_argument("--proco-temperature", type=float, default=0.07, help="ProCo temperature τ")
    parser.add_argument(
        "--proco-lambda-uncertainty",
        type=float,
        default=0.1,
        help="ProCo 不确定性惩罚系数 λ_u",
    )
    parser.add_argument("--no-missing-augmentation", action="store_true", help="训练时不做缺失增强")
    parser.add_argument("--no-class-balance", action="store_true", help="不使用少数类重采样")
    parser.add_argument("--n-folds", type=int, default=5, help="实际训练的折数，1 表示只跑第 0 折")
    return parser.parse_args()


def _ablation_name(exp_name: str) -> Optional[str]:
    if "ours_" not in exp_name:
        return None
    parts = exp_name.split("_", 1)
    if len(parts) != 2 or parts[1] == "proco":
        return None
    return parts[1].replace("proco_", "")


def main():
    args = parse_args()
    if args.list_models:
        print("可选模型/实验：")
        for name in list_models():
            print("  -", name)
        return

    if not args.experiment:
        raise SystemExit("请指定 --experiment，或使用 --list-models 查看名称。")
    if not args.ct_root or not args.clinical_xlsx:
        raise SystemExit(
            "请通过 --ct-root 和 --clinical-xlsx 指定数据路径。\n"
            "示例: python main.py --experiment ours_proco --ct-root /path/to/ct --clinical-xlsx /path/to/clinical.xlsx"
        )

    exp_name = args.experiment
    available = set(list_models())
    if exp_name not in available:
        raise SystemExit(
            f"未知 experiment '{exp_name}'.\n"
            + "\n".join(f"  - {m}" for m in sorted(available))
        )

    use_proco = exp_name.startswith("ours_proco") or exp_name.endswith("_proco")
    class2idx, idx2class, num_classes = get_class_maps(args.three_class)
    out_root = Path(args.out_root) / exp_name
    cfg = ExperimentConfig(
        ct_root=args.ct_root,
        clinical_xlsx=args.clinical_xlsx,
        out_root=out_root,
        use_3_class=args.three_class,
        model_name=exp_name,
        use_proco=use_proco,
        ablation_name=_ablation_name(exp_name),
        lambda_proco=args.lambda_proco,
        proco_temperature=args.proco_temperature,
        proco_lambda_uncertainty=args.proco_lambda_uncertainty,
        use_missing_augmentation=not args.no_missing_augmentation,
        use_class_balance=not args.no_class_balance,
        class2idx=class2idx,
        idx2class=idx2class,
        n_folds=args.n_folds,
    )

    print(f"[Experiment] {exp_name}")
    print(f"  use_proco={cfg.use_proco}, ablation={cfg.ablation_name}, n_folds={cfg.n_folds}")
    print(f"  out_root={cfg.out_root}")
    print(f"  classes={idx2class}")

    fold_metrics = run_experiment(cfg, num_classes=num_classes, idx2class=idx2class)
    accs = [m["acc"] for m in fold_metrics]
    baccs = [m["balanced_acc"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]
    aucs = [m["auc_macro"] for m in fold_metrics]
    print(f"\n=== {len(fold_metrics)}-fold Summary ===")
    print(f"AUC      : {sum(aucs) / len(aucs):.3f}")
    print(f"Macro-F1 : {sum(f1s) / len(f1s):.3f}")
    print(f"Acc      : {sum(accs) / len(accs):.3f}")
    print(f"BAcc     : {sum(baccs) / len(baccs):.3f}")


if __name__ == "__main__":
    main()
