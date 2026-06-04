from pathlib import Path
import csv
import shutil
import sys


# ============================================================
# 1. 路径设置
# ============================================================

# PyMOL 筛选结果文件
CSV_FILE = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3_tiaoxuan\9IR\9IR_pymol_all_ranking_fixed.csv"
)

# 原始 CIF 文件夹
CIF_SOURCE_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3_tiaoxuan\9IR\9IR_CIF"
)

# 合格候选骨架输出文件夹
CIF_OUTPUT_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3_tiaoxuan\9IR\9IR_candidate_CIF"
)

# 提取结果清单
MANIFEST_FILE = CIF_OUTPUT_DIR / "9IR_candidate_CIF_manifest.csv"


# ============================================================
# 2. 筛选参数
# ============================================================

GRADE_COLUMN = "pymol_grade"
FILE_COLUMN = "json_file_name"

PASS_GRADES = {
    "Excellent_hydrophobic_pocket",
    "Good_hydrophobic_pocket",
}

EXPECTED_TOTAL = 75
EXPECTED_EXCELLENT = 61
EXPECTED_GOOD = 14


# ============================================================
# 3. 主程序
# ============================================================

def extract_candidate_cif():
    """根据 PyMOL 评分结果，提取合格的 9IR 候选 CIF 骨架文件。"""

    # ---------- 检查输入文件与文件夹 ----------
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"未找到 PyMOL 排名文件：\n{CSV_FILE}")

    if not CIF_SOURCE_DIR.exists():
        raise FileNotFoundError(f"未找到 CIF 文件夹：\n{CIF_SOURCE_DIR}")

    CIF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    excellent_count = 0
    good_count = 0

    # ---------- 读取 CSV 并筛选合格模型 ----------
    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV 文件为空，或无法读取表头。")

        # 去除表头可能存在的前后空格
        reader.fieldnames = [col.strip() for col in reader.fieldnames]

        required_columns = {GRADE_COLUMN, FILE_COLUMN}
        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"CSV 文件缺少必要列：{sorted(missing_columns)}\n"
                f"当前检测到的列为：{reader.fieldnames}"
            )

        for row in reader:
            row = {str(k).strip(): str(v).strip() for k, v in row.items()}

            grade = row.get(GRADE_COLUMN, "")
            json_file_name = row.get(FILE_COLUMN, "")

            if grade in PASS_GRADES and json_file_name:
                selected_rows.append(row)

                if grade == "Excellent_hydrophobic_pocket":
                    excellent_count += 1
                elif grade == "Good_hydrophobic_pocket":
                    good_count += 1

    # ---------- 输出筛选概况 ----------
    print("=" * 75)
    print("9IR candidate CIF extraction")
    print("=" * 75)
    print(f"输入 CSV 文件：{CSV_FILE}")
    print(f"原始 CIF 文件夹：{CIF_SOURCE_DIR}")
    print(f"输出文件夹：{CIF_OUTPUT_DIR}")
    print()
    print("筛选条件：")
    print("  pymol_grade == Excellent_hydrophobic_pocket")
    print("  或")
    print("  pymol_grade == Good_hydrophobic_pocket")
    print()
    print(f"Excellent_hydrophobic_pocket 数量：{excellent_count}")
    print(f"Good_hydrophobic_pocket 数量：{good_count}")
    print(f"合格候选模型总数：{len(selected_rows)}")
    print()

    # ---------- 根据 json_file_name 复制对应 CIF ----------
    manifest_records = []
    copied_files = set()

    copied_count = 0
    missing_count = 0
    duplicate_count = 0

    for index, row in enumerate(selected_rows, start=1):
        grade = row[GRADE_COLUMN]
        json_file_name = row[FILE_COLUMN]

        # 文件名转换规则：
        # 9IR_9IR_buried_5_model_6.json -> 9IR_9IR_buried_5_model_6.cif
        cif_file_name = Path(json_file_name).with_suffix(".cif").name

        source_cif = CIF_SOURCE_DIR / cif_file_name
        target_cif = CIF_OUTPUT_DIR / cif_file_name

        status = ""

        if cif_file_name in copied_files:
            status = "Duplicate_skipped"
            duplicate_count += 1
            print(f"[重复跳过] {cif_file_name}")

        elif source_cif.exists():
            shutil.copy2(source_cif, target_cif)
            copied_files.add(cif_file_name)
            copied_count += 1
            status = "Copied"

            print(
                f"[{index:>2}/{len(selected_rows)}] 已复制："
                f"{cif_file_name} | {grade}"
            )

        else:
            missing_count += 1
            status = "Missing_source_CIF"

            print(f"[缺失] 未找到对应 CIF 文件：{cif_file_name}")

        manifest_records.append(
            {
                "selection_rank": index,
                "pymol_grade": grade,
                "json_file_name": json_file_name,
                "cif_file_name": cif_file_name,
                "source_cif": str(source_cif),
                "output_cif": str(target_cif),
                "copy_status": status,
            }
        )

    # ---------- 保存提取清单 ----------
    manifest_fields = [
        "selection_rank",
        "pymol_grade",
        "json_file_name",
        "cif_file_name",
        "source_cif",
        "output_cif",
        "copy_status",
    ]

    with MANIFEST_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_records)

    # ---------- 最终汇总 ----------
    print()
    print("=" * 75)
    print("提取完成")
    print("=" * 75)
    print(f"Excellent_hydrophobic_pocket：{excellent_count}")
    print(f"Good_hydrophobic_pocket：{good_count}")
    print(f"合格候选总数：{len(selected_rows)}")
    print(f"成功复制 CIF 文件数：{copied_count}")
    print(f"重复跳过文件数：{duplicate_count}")
    print(f"未找到 CIF 文件数：{missing_count}")
    print()
    print(f"候选 CIF 输出文件夹：{CIF_OUTPUT_DIR}")
    print(f"提取记录表：{MANIFEST_FILE}")

    # ---------- 与当前筛选结果进行数量核对 ----------
    print()
    print("=" * 75)
    print("数量核对")
    print("=" * 75)

    if (
        len(selected_rows) == EXPECTED_TOTAL
        and excellent_count == EXPECTED_EXCELLENT
        and good_count == EXPECTED_GOOD
        and copied_count == EXPECTED_TOTAL
        and missing_count == 0
    ):
        print("结果检查通过：")
        print("  已成功提取 75 个合格 9IR 候选骨架。")
        print("  其中 Excellent_hydrophobic_pocket = 61 个。")
        print("  其中 Good_hydrophobic_pocket = 14 个。")
    else:
        print("当前结果与预期数量不完全一致，请检查：")
        print(f"  预期候选总数：{EXPECTED_TOTAL}；当前识别：{len(selected_rows)}")
        print(f"  预期 Excellent：{EXPECTED_EXCELLENT}；当前识别：{excellent_count}")
        print(f"  预期 Good：{EXPECTED_GOOD}；当前识别：{good_count}")
        print(f"  当前成功复制 CIF 数量：{copied_count}")
        print(f"  当前缺失 CIF 数量：{missing_count}")


if __name__ == "__main__":
    try:
        extract_candidate_cif()
    except Exception as exc:
        print("\n程序运行失败：")
        print(exc)
        sys.exit(1)