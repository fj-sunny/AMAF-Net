"""
统一主入口：AMAF-Net 模型选择 + 训练 / 测试

用法示例（在项目根目录运行，数据路径需自行指定）::

    python -m amaf_net.main --experiment ours_proco \\
        --ct-root /path/to/your/ct_roi \\
        --clinical-xlsx /path/to/your/clinical.xlsx \\
        --out-root ./output/amaf_ours_proco

    python -m amaf_net.main --list-models  # 列出所有可选模型
"""

from __future__ import annotations

import argparse
from pathlib import Path

from train_missing_aware_fusion import (  # type: ignore
    USE_3_CLASS,
    CLASS2IDX,
    IDX2CLASS,
    NUM_CLASSES,
)

from .models.registry import list_models
from .train.trainer import ExperimentConfig, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Missing-aware multimodal fusion unified runner")
    parser.add_argument(
        "--ct-root",
        type=str,
        default="",
        help="NC ROI CT 根目录（必填，请勿在代码中写死路径）",
    )
    parser.add_argument(
        "--clinical-xlsx",
        type=str,
        default="",
        help="临床 Excel 文件路径（必填，请勿在代码中写死路径）",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default="./output/amaf_net",
        help="输出根目录",
    )
    parser.add_argument(
        "--three-class",
        action="store_true",
        default=USE_3_CLASS,
        help="是否使用 3 类 (CS/PA/PC) 设置；默认与原脚本一致",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help=(
            "实验/模型名称（见 --list-models），如：\n"
            "  image_resnet18, image_unet3d, image_vit,\n"
            "  clinical_mlp, clinical_mlp_missing, clinical_ft_transformer,\n"
            "  fusion_concat, fusion_cross_attn, fusion_late, fusion_daft,\n"
            "  fusion_hyperfusion, fusion_drfuse,\n"
            "  ours, ours_proco, ours_A1, ours_proco_A1, ..."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="列出所有可选模型名称后退出",
    )
    parser.add_argument(
        "--lambda-proco",
        type=float,
        default=0.1,
        help="ProCo 损失权重 λ_ProCo，仅对 *_proco 实验有效",
    )
    parser.add_argument(
        "--proco-temperature",
        type=float,
        default=0.07,
        help="ProCo temperature 参数 τ，仅对 *_proco 实验有效",
    )
    parser.add_argument(
        "--proco-lambda-uncertainty",
        type=float,
        default=0.1,
        help="ProCo 不确定性惩罚系数 λ_u，仅对 *_proco 实验有效",
    )
    parser.add_argument(
        "--no-missing-augmentation",
        action="store_true",
        help="训练时不做缺失增强（对应 D2 消融思想）",
    )
    parser.add_argument(
        "--no-class-balance",
        action="store_true",
        help="不使用类别重采样（对应 D1 消融思想）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.list_models and (not args.ct_root or not args.clinical_xlsx):
        raise SystemExit(
            "请通过 --ct-root 和 --clinical-xlsx 指定数据路径，勿在仓库中提交真实路径。\n"
            "示例: python -m amaf_net.main --experiment ours_proco --ct-root /path/to/ct --clinical-xlsx /path/to/clinical.xlsx"
        )

    if args.list_models:
        print("可选模型/实验：")
        for name in list_models():
            print("  -", name)
        return

    exp_name: str = args.experiment
    available = set(list_models())
    if exp_name not in available:
        raise SystemExit(
            f"未知 experiment '{exp_name}'.\n"
            f"可选模型名称如下（或使用 --list-models 查看）：\n"
            + "\n".join(f"  - {m}" for m in sorted(available))
        )

    # 解析是否为 ProCo 实验
    use_proco = exp_name.startswith("ours_proco") or exp_name.endswith("_proco")
    # 从名称中提取 ablation（例如 ours_A1 / ours_proco_A1）
    ablation_name = None
    if "ours_" in exp_name:
        parts = exp_name.split("_", 1)
        if len(parts) == 2 and parts[1] not in ("proco",):
            # ours_A1 / ours_proco_A1
            ablation_name = parts[1].replace("proco_", "")

    out_root = Path(args.out_root) / exp_name

    cfg = ExperimentConfig(
        ct_root=args.ct_root,
        clinical_xlsx=args.clinical_xlsx,
        out_root=out_root,
        use_3_class=args.three_class,
        model_name=exp_name,
        use_proco=use_proco,
        ablation_name=ablation_name,
        lambda_proco=args.lambda_proco,
        proco_temperature=args.proco_temperature,
        proco_lambda_uncertainty=args.proco_lambda_uncertainty,
        use_missing_augmentation=not args.no_missing_augmentation,
        use_class_balance=not args.no_class_balance,
    )

    # 根据 3 类 / 4 类设置 label 映射
    if args.three_class:
        class2idx = {"CS": 0, "PA": 1, "PC": 2}
    else:
        class2idx = {"CS": 0, "PA": 1, "PC": 2, "NFA": 3}
    idx2class = {v: k for k, v in class2idx.items()}
    num_classes = len(class2idx)

    print(f"[Experiment] {exp_name}")
    print(f"  use_proco={cfg.use_proco}, ablation={cfg.ablation_name}")
    print(f"  out_root={cfg.out_root}")
    print(f"  classes={idx2class}")

    fold_metrics = run_experiment(cfg, num_classes=num_classes, idx2class=idx2class)

    # 简要汇总核心指标，完整详细结果仍由底层 train_* 函数在 out_root 下落盘
    accs = [m["acc"] for m in fold_metrics]
    baccs = [m["balanced_acc"] for m in fold_metrics]
    f1s = [m["macro_f1"] for m in fold_metrics]
    aucs = [m["auc_macro"] for m in fold_metrics]
    print("\n=== 5-fold Summary (简要) ===")
    print(f"AUC↑      : {sum(aucs)/len(aucs):.3f}")
    print(f"Macro-F1↑ : {sum(f1s)/len(f1s):.3f}")
    print(f"Acc↑      : {sum(accs)/len(accs):.3f}")
    print(f"BAcc↑     : {sum(baccs)/len(baccs):.3f}")


if __name__ == "__main__":
    main()

