from pathlib import Path
import csv
import shutil
import sys


# ============================================================
# 1. 路径设置
# ============================================================

# PyMOL 筛选结果 CSV 文件
CSV_FILE = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3_tiaoxuan\4DY\4DY_pymol_all_ranking_fixed.csv"
)

# 原始 CIF 文件所在文件夹
CIF_SOURCE_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3_tiaoxuan\4DY\4DY_CIF"
)

# 输出文件夹：保存被评定为 Good_pocket 的候选 CIF 文件
CIF_OUTPUT_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3_tiaoxuan\4DY\4DY_Good_pocket_CIF"
)

# 输出清单文件
MANIFEST_FILE = CIF_OUTPUT_DIR / "4DY_Good_pocket_CIF_manifest.csv"


# ============================================================
# 2. 参数设置
# ============================================================

GRADE_COLUMN = "pymol_grade"
FILE_COLUMN = "json_file_name"
TARGET_GRADE = "Good_pocket"


# ============================================================
# 3. 主程序
# ============================================================

def extract_good_pocket_cif():
    """根据 CSV 中的 Good_pocket 标记，复制对应的 CIF 文件。"""

    # ---------- 检查输入路径 ----------
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"未找到 CSV 文件：\n{CSV_FILE}")

    if not CIF_SOURCE_DIR.exists():
        raise FileNotFoundError(f"未找到 CIF 文件夹：\n{CIF_SOURCE_DIR}")

    # 创建输出文件夹
    CIF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    copied_records = []
    missing_records = []

    # ---------- 读取 CSV 并筛选 Good_pocket ----------
    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV 文件为空，或无法识别表头。")

        required_columns = {GRADE_COLUMN, FILE_COLUMN}
        missing_columns = required_columns - set(reader.fieldnames)

        if missing_columns:
            raise ValueError(
                f"CSV 文件缺少必要列：{missing_columns}\n"
                f"当前检测到的列为：{reader.fieldnames}"
            )

        for row in reader:
            grade = str(row.get(GRADE_COLUMN, "")).strip()
            json_file_name = str(row.get(FILE_COLUMN, "")).strip()

            if grade == TARGET_GRADE and json_file_name:
                selected_rows.append(row)

    print("=" * 70)
    print("4DY Good_pocket CIF extraction")
    print("=" * 70)
    print(f"CSV 文件：{CSV_FILE}")
    print(f"原始 CIF 文件夹：{CIF_SOURCE_DIR}")
    print(f"输出文件夹：{CIF_OUTPUT_DIR}")
    print(f"筛选条件：{GRADE_COLUMN} == {TARGET_GRADE}")
    print(f"筛选得到候选模型数：{len(selected_rows)}")
    print()

    # ---------- 根据 JSON 文件名寻找对应 CIF 文件 ----------
    copied_names = set()

    for index, row in enumerate(selected_rows, start=1):
        json_file_name = str(row[FILE_COLUMN]).strip()

        # 规则：xxx.json -> xxx.cif
        cif_file_name = Path(json_file_name).with_suffix(".cif").name

        source_cif = CIF_SOURCE_DIR / cif_file_name
        output_cif = CIF_OUTPUT_DIR / cif_file_name

        record = {
            "selection_rank": index,
            "pymol_grade": row.get(GRADE_COLUMN, ""),
            "json_file_name": json_file_name,
            "cif_file_name": cif_file_name,
            "source_cif": str(source_cif),
            "output_cif": str(output_cif),
            "copy_status": "",
        }

        # 防止 CSV 中重复记录导致重复复制
        if cif_file_name in copied_names:
            record["copy_status"] = "Duplicate_skipped"
            copied_records.append(record)
            print(f"[重复跳过] {cif_file_name}")
            continue

        if source_cif.exists():
            shutil.copy2(source_cif, output_cif)
            copied_names.add(cif_file_name)

            record["copy_status"] = "Copied"
            copied_records.append(record)

            print(f"[{index:>2}/{len(selected_rows)}] 已复制：{cif_file_name}")
        else:
            record["copy_status"] = "Missing_source_CIF"
            missing_records.append(record)

            print(f"[缺失] 未找到：{cif_file_name}")

    # ---------- 输出提取清单 ----------
    all_records = copied_records + missing_records

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
        writer.writerows(all_records)

    copied_count = sum(r["copy_status"] == "Copied" for r in copied_records)
    duplicate_count = sum(r["copy_status"] == "Duplicate_skipped" for r in copied_records)
    missing_count = len(missing_records)

    # ---------- 输出结果汇总 ----------
    print()
    print("=" * 70)
    print("提取完成")
    print("=" * 70)
    print(f"Good_pocket 记录数：{len(selected_rows)}")
    print(f"成功复制 CIF 数量：{copied_count}")
    print(f"重复跳过数量：{duplicate_count}")
    print(f"未找到 CIF 数量：{missing_count}")
    print(f"候选 CIF 输出文件夹：{CIF_OUTPUT_DIR}")
    print(f"提取清单文件：{MANIFEST_FILE}")

    if missing_count > 0:
        print()
        print("以下 CIF 文件未找到，请检查文件名或原始 CIF 文件夹：")
        for record in missing_records:
            print(f"  - {record['cif_file_name']}")

    # 按你的 PyMOL 结果，4DY 理论上应有 31 个 Good_pocket 模型
    expected_count = 31
    if len(selected_rows) == expected_count and copied_count == expected_count:
        print()
        print("结果检查通过：已成功提取预期的 31 个 Good_pocket CIF 骨架。")
    else:
        print()
        print(
            f"提示：根据当前筛选结果，理论预期约为 {expected_count} 个 Good_pocket 模型；"
            f"当前识别 {len(selected_rows)} 个，成功复制 {copied_count} 个。"
        )


if __name__ == "__main__":
    try:
        extract_good_pocket_cif()
    except Exception as exc:
        print("\n程序运行失败：")
        print(exc)
        sys.exit(1)