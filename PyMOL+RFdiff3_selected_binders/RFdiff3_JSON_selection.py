import argparse
import json
import math
from pathlib import Path

import pandas as pd


# ============================================================
# 1. 默认路径
# ============================================================

DEFAULT_INPUT_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3\4DY_output\all\JSON"
)


# ============================================================
# 2. 工具函数
# ============================================================

def safe_get(dictionary, key, default=0):
    """安全读取字典中的值，避免 None 或缺失字段导致报错。"""
    value = dictionary.get(key, default)
    if value is None:
        return default
    return value


def get_task_name(data, json_file):
    """从 RFdiffusion3 JSON 中提取任务名称。"""
    specification = data.get("specification", {})
    extra = specification.get("extra", {})

    task_name = extra.get("task_name", "")
    example = extra.get("example", "")

    if task_name:
        return task_name
    if example:
        return example

    return json_file.stem


def calculate_json_score(
    extra_chainbreaks,
    backbone_clashes,
    sidechain_clashes,
    ligand_clashes,
    ligand_min_distance,
    max_ca_deviation,
    loop_fraction,
    non_loop_fraction,
    radius_of_gyration,
    num_residues,
    alanine_content,
    glycine_content,
):
    """
    RFdiffusion3 JSON 初筛综合评分。
    分数越低越好。

    这个分数只用于第一轮筛选，不能替代 PyMOL 人工/自动检查。
    """

    if num_residues and num_residues > 0:
        rg_norm = radius_of_gyration / (num_residues ** (1 / 3))
    else:
        rg_norm = 999

    ligand_distance_penalty = 0

    if ligand_min_distance < 2.2:
        ligand_distance_penalty = 1000
    elif ligand_min_distance < 2.8:
        ligand_distance_penalty = 300
    elif ligand_min_distance < 3.0:
        ligand_distance_penalty = 80

    score = (
        1000 * extra_chainbreaks
        + 1000 * backbone_clashes
        + 800 * ligand_clashes
        + 300 * sidechain_clashes
        + ligand_distance_penalty
        + 10 * max_ca_deviation
        + 80 * loop_fraction
        + 40 * max(0, 0.35 - non_loop_fraction)
        + 20 * max(0, rg_norm - 3.8)
        + 100 * max(0, alanine_content - 0.20)
        + 100 * max(0, glycine_content - 0.15)
    )

    return score, rg_norm


def classify_json_candidate(
    extra_chainbreaks,
    backbone_clashes,
    sidechain_clashes,
    ligand_clashes,
    ligand_min_distance,
    max_ca_deviation,
    loop_fraction,
    non_loop_fraction,
    radius_of_gyration,
):
    """根据 RFdiffusion3 JSON 指标进行初步分级。"""

    if extra_chainbreaks > 0:
        return "Poor_extra_chainbreak"

    if backbone_clashes > 0:
        return "Poor_backbone_clash"

    if ligand_clashes > 0:
        return "Poor_ligand_clash"

    if ligand_min_distance < 2.2:
        return "Poor_ligand_overlap"

    if sidechain_clashes >= 5:
        return "Poor_many_sidechain_clashes"

    if sidechain_clashes > 0:
        return "Medium_sidechain_clash"

    if (
        max_ca_deviation <= 1.0
        and loop_fraction <= 0.45
        and non_loop_fraction >= 0.50
        and radius_of_gyration <= 25
    ):
        return "Good"

    if (
        max_ca_deviation <= 1.5
        and loop_fraction <= 0.60
        and non_loop_fraction >= 0.35
        and radius_of_gyration <= 30
    ):
        return "Acceptable"

    if loop_fraction > 0.65:
        return "Need_manual_check_loop_rich"

    if radius_of_gyration > 30:
        return "Need_manual_check_large_Rg"

    return "Need_manual_check"


def visual_priority_from_json(
    extra_chainbreaks,
    backbone_clashes,
    sidechain_clashes,
    ligand_clashes,
    ligand_min_distance,
    max_ca_deviation,
    loop_fraction,
    non_loop_fraction,
):
    """根据 JSON 指标决定是否优先进入 PyMOL 检查。"""

    if (
        extra_chainbreaks == 0
        and backbone_clashes == 0
        and ligand_clashes == 0
        and sidechain_clashes == 0
        and ligand_min_distance >= 3.0
        and max_ca_deviation <= 1.0
        and loop_fraction <= 0.60
        and non_loop_fraction >= 0.35
    ):
        return "High_priority_visual_check"

    if (
        extra_chainbreaks == 0
        and backbone_clashes == 0
        and ligand_clashes == 0
        and ligand_min_distance >= 2.8
        and max_ca_deviation <= 1.5
    ):
        return "Medium_priority_visual_check"

    return "Low_priority"


# ============================================================
# 3. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prefilter RFdiffusion3 small-molecule binder JSON outputs."
    )

    parser.add_argument(
        "--input_dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing RFdiffusion3 JSON files.",
    )

    parser.add_argument(
        "--output_dir",
        default="",
        help="Directory for output CSV files. Default: input_dir/ranking_json",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / "ranking_json"

    output_dir.mkdir(parents=True, exist_ok=True)

    all_csv = output_dir / "rfd3_json_all.csv"
    passed_csv = output_dir / "rfd3_json_pass.csv"
    top10_csv = output_dir / "rfd3_json_top10.csv"
    top20_csv = output_dir / "rfd3_json_top20.csv"

    task_top10_dir = output_dir / "top10_by_task"
    task_top10_dir.mkdir(exist_ok=True)

    json_files = sorted(input_dir.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    records = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Warning] Cannot read {json_file.name}: {e}")
            continue

        metrics = data.get("metrics", {})
        specification = data.get("specification", {})
        extra = specification.get("extra", {})

        ligand = specification.get("ligand", "")
        task_name = get_task_name(data, json_file)
        example_id = extra.get("example_id", json_file.stem)
        example = extra.get("example", "")

        input_structure = specification.get("input", "")

        # 对小分子 binder，一般不默认允许 chainbreak
        expected_chainbreaks = 0

        n_chainbreaks = safe_get(metrics, "n_chainbreaks", 999)
        extra_chainbreaks = max(0, n_chainbreaks - expected_chainbreaks)

        backbone_clashes = safe_get(
            metrics,
            "n_clashing.interresidue_clashes_w_backbone",
            999,
        )

        sidechain_clashes = safe_get(
            metrics,
            "n_clashing.interresidue_clashes_w_sidechain",
            999,
        )

        ligand_clashes = safe_get(
            metrics,
            "n_clashing.ligand_clashes",
            999,
        )

        ligand_min_distance = safe_get(
            metrics,
            "n_clashing.ligand_min_distance",
            0,
        )

        max_ca_deviation = safe_get(metrics, "max_ca_deviation", 999)
        loop_fraction = safe_get(metrics, "loop_fraction", 1)
        non_loop_fraction = safe_get(metrics, "non_loop_fraction", 0)
        helix_fraction = safe_get(metrics, "helix_fraction", 0)
        sheet_fraction = safe_get(metrics, "sheet_fraction", 0)
        num_ss_elements = safe_get(metrics, "num_ss_elements", 0)
        radius_of_gyration = safe_get(metrics, "radius_of_gyration", 999)
        alanine_content = safe_get(metrics, "alanine_content", 0)
        glycine_content = safe_get(metrics, "glycine_content", 0)
        num_residues = safe_get(metrics, "num_residues", 0)

        json_score, rg_norm = calculate_json_score(
            extra_chainbreaks=extra_chainbreaks,
            backbone_clashes=backbone_clashes,
            sidechain_clashes=sidechain_clashes,
            ligand_clashes=ligand_clashes,
            ligand_min_distance=ligand_min_distance,
            max_ca_deviation=max_ca_deviation,
            loop_fraction=loop_fraction,
            non_loop_fraction=non_loop_fraction,
            radius_of_gyration=radius_of_gyration,
            num_residues=num_residues,
            alanine_content=alanine_content,
            glycine_content=glycine_content,
        )

        json_grade = classify_json_candidate(
            extra_chainbreaks=extra_chainbreaks,
            backbone_clashes=backbone_clashes,
            sidechain_clashes=sidechain_clashes,
            ligand_clashes=ligand_clashes,
            ligand_min_distance=ligand_min_distance,
            max_ca_deviation=max_ca_deviation,
            loop_fraction=loop_fraction,
            non_loop_fraction=non_loop_fraction,
            radius_of_gyration=radius_of_gyration,
        )

        visual_priority = visual_priority_from_json(
            extra_chainbreaks=extra_chainbreaks,
            backbone_clashes=backbone_clashes,
            sidechain_clashes=sidechain_clashes,
            ligand_clashes=ligand_clashes,
            ligand_min_distance=ligand_min_distance,
            max_ca_deviation=max_ca_deviation,
            loop_fraction=loop_fraction,
            non_loop_fraction=non_loop_fraction,
        )

        select_buried = specification.get("select_buried", {}).get(ligand, "")
        select_exposed = specification.get("select_exposed", {}).get(ligand, "")

        # JSON 初筛通过条件：只基于 JSON 指标，不要求结构文件在同一目录
        json_pass = (
            extra_chainbreaks == 0
            and backbone_clashes == 0
            and ligand_clashes == 0
            and ligand_min_distance >= 2.8
            and max_ca_deviation <= 2.0
            and loop_fraction <= 0.70
            and radius_of_gyration <= 30
        )

        records.append(
            {
                "json_file": str(json_file),
                "json_file_name": json_file.name,
                "example_id": example_id,
                "example": example,
                "task_name": task_name,
                "ligand": ligand,
                "input_structure": input_structure,

                "json_score": round(json_score, 4),
                "json_pass": json_pass,
                "json_grade": json_grade,
                "visual_priority": visual_priority,

                "n_chainbreaks_raw": n_chainbreaks,
                "expected_chainbreaks": expected_chainbreaks,
                "extra_chainbreaks": extra_chainbreaks,
                "backbone_clashes": backbone_clashes,
                "sidechain_clashes": sidechain_clashes,
                "ligand_clashes": ligand_clashes,
                "ligand_min_distance": ligand_min_distance,
                "max_ca_deviation": max_ca_deviation,
                "radius_of_gyration": radius_of_gyration,
                "rg_norm": round(rg_norm, 4) if math.isfinite(rg_norm) else rg_norm,
                "loop_fraction": loop_fraction,
                "non_loop_fraction": non_loop_fraction,
                "helix_fraction": helix_fraction,
                "sheet_fraction": sheet_fraction,
                "num_ss_elements": num_ss_elements,
                "alanine_content": alanine_content,
                "glycine_content": glycine_content,
                "num_residues": num_residues,

                "buried_atoms": select_buried,
                "exposed_atoms": select_exposed,

                # 这些列留给后续 PyMOL 自动/人工检查填写
                "ligand_present": "",
                "ligand_complete": "",
                "buried_contact": "",
                "exposed_contact": "",
                "pocket_or_groove": "",
                "severe_ligand_clash_manual": "",
                "manual_priority": "",
                "notes": "",
            }
        )

    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError("No valid JSON records were parsed.")

    # 排序：先看是否通过，再看分数
    df = df.sort_values(
        by=[
            "json_pass",
            "json_score",
            "extra_chainbreaks",
            "backbone_clashes",
            "ligand_clashes",
            "sidechain_clashes",
            "max_ca_deviation",
            "loop_fraction",
            "radius_of_gyration",
        ],
        ascending=[
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    df.insert(0, "json_rank", df.index + 1)

    passed_df = df[df["json_pass"]].copy()

    # 输出总表和通过表
    df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    passed_df.to_csv(passed_csv, index=False, encoding="utf-8-sig")
    df.head(10).to_csv(top10_csv, index=False, encoding="utf-8-sig")
    df.head(20).to_csv(top20_csv, index=False, encoding="utf-8-sig")

    # 按 task 输出 top10
    for task, sub_df in df.groupby("task_name"):
        safe_task = str(task).replace("/", "_").replace("\\", "_").replace(" ", "_")
        task_csv = task_top10_dir / f"top10_{safe_task}.csv"
        sub_df.head(10).to_csv(task_csv, index=False, encoding="utf-8-sig")

    # 打印关键结果
    important_cols = [
        "json_rank",
        "json_file_name",
        "example_id",
        "task_name",
        "ligand",
        "json_score",
        "json_pass",
        "json_grade",
        "visual_priority",
        "n_chainbreaks_raw",
        "extra_chainbreaks",
        "backbone_clashes",
        "sidechain_clashes",
        "ligand_clashes",
        "ligand_min_distance",
        "max_ca_deviation",
        "radius_of_gyration",
        "loop_fraction",
        "non_loop_fraction",
        "helix_fraction",
        "sheet_fraction",
    ]

    print("\n========== RFdiffusion3 JSON prefilter ==========\n")
    print(f"Input directory: {input_dir}")
    print(f"Total JSON files read: {len(df)}")
    print(f"Passed JSON filter: {len(passed_df)}")

    print("\n========== Top 10 by JSON score ==========\n")
    print(df[important_cols].head(10).to_string(index=False))

    print("\n========== Grade Counts ==========\n")
    print(df["json_grade"].value_counts().to_string())

    print("\n========== Visual Priority Counts ==========\n")
    print(df["visual_priority"].value_counts().to_string())

    print("\n========== Output Files ==========\n")
    print(f"All ranking saved to: {all_csv}")
    print(f"Passed candidates saved to: {passed_csv}")
    print(f"Top 10 saved to: {top10_csv}")
    print(f"Top 20 saved to: {top20_csv}")
    print(f"Task-specific top10 saved to: {task_top10_dir}")

    print("\n========== Recommended Next Step ==========\n")
    print("1. Open rfd3_json_pass.csv.")
    print("2. Locate the corresponding CIF/PDB structures in your structure folder.")
    print("3. Use PyMOL to check whether chain A forms a pocket or groove around chain B ligand.")
    print("4. Check buried_atom contact and exposed_atom partial exposure.")
    print("5. Select top candidates for ProteinMPNN sequence design.")


if __name__ == "__main__":
    main()