from pathlib import Path
import csv
import shutil
import sys


# ============================================================
# 1. 输入与输出路径
# ============================================================

# PyMOL 精筛结果 CSV 文件
CSV_FILE = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3_tiaoxuan\QUE\QUE_pymol_all_mixed_ranking.csv"
)

# RFdiffusion3 生成的全部 QUE CIF 骨架所在文件夹
CIF_SOURCE_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3_tiaoxuan\QUE\QUE_CIF"
)

# 所有通过 PyMOL 精筛的候选 CIF 输出文件夹
CIF_OUTPUT_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3_tiaoxuan\QUE\QUE_candidate_CIF"
)

# 按类型分别保存的子文件夹
EXCELLENT_BURIED_DIR = CIF_OUTPUT_DIR / "Excellent_QUE_buried_pocket"
GOOD_BURIED_DIR = CIF_OUTPUT_DIR / "Good_QUE_buried_pocket"
GOOD_PARTIAL_DIR = CIF_OUTPUT_DIR / "Good_QUE_partial_pocket"

# 候选骨架提取记录表
MANIFEST_FILE = CIF_OUTPUT_DIR / "QUE_candidate_CIF_manifest.csv"


# ============================================================
# 2. 筛选设置
# ============================================================

PASS_GRADES = {
    "Excellent_QUE_buried_pocket",
    "Good_QUE_buried_pocket",
    "Good_QUE_partial_pocket",
}

# 兼容你描述的列名，也兼容脚本可能输出的简写列名
POSSIBLE_GRADE_COLUMNS = ["pymol_grade", "grade"]
POSSIBLE_FILE_COLUMNS = ["json_file_name", "json_file"]


# ============================================================
# 3. 工具函数
# ============================================================

def find_column(fieldnames, possible_names):
    """
    在 CSV 表头中寻找可用列名。
    支持例如 pymol_grade / grade 或 json_file_name / json_file。
    """
    normalized = {name.strip(): name for name in fieldnames}

    for candidate in possible_names:
        if candidate in normalized:
            return normalized[candidate]

    raise ValueError(
        f"未找到所需列。可接受的列名为：{possible_names}\n"
        f"当前 CSV 中检测到的列为：{fieldnames}"
    )


def get_grade_subfolder(grade):
    """根据 PyMOL 评价等级返回对应输出子文件夹。"""
    if grade == "Excellent_QUE_buried_pocket":
        return EXCELLENT_BURIED_DIR
    if grade == "Good_QUE_buried_pocket":
        return GOOD_BURIED_DIR
    if grade == "Good_QUE_partial_pocket":
        return GOOD_PARTIAL_DIR

    raise ValueError(f"未知的合格等级：{grade}")


# ============================================================
# 4. 主程序
# ============================================================

def extract_que_candidate_cif():
    """根据 PyMOL 筛选结果，提取合格的 QUE 候选 CIF 骨架。"""

    # ---------- 检查输入路径 ----------
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"未找到 PyMOL 结果文件：\n{CSV_FILE}")

    if not CIF_SOURCE_DIR.exists():
        raise FileNotFoundError(f"未找到原始 CIF 文件夹：\n{CIF_SOURCE_DIR}")

    # ---------- 创建输出文件夹 ----------
    CIF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXCELLENT_BURIED_DIR.mkdir(parents=True, exist_ok=True)
    GOOD_BURIED_DIR.mkdir(parents=True, exist_ok=True)
    GOOD_PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

    selected_rows = []

    # ---------- 读取 CSV 并筛选合格模型 ----------
    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV 文件为空，或无法识别表头。")

        fieldnames = [name.strip() for name in reader.fieldnames]

        grade_column = find_column(fieldnames, POSSIBLE_GRADE_COLUMNS)
        file_column = find_column(fieldnames, POSSIBLE_FILE_COLUMNS)

        for row in reader:
            clean_row = {
                str(key).strip(): str(value).strip()
                for key, value in row.items()
            }

            grade = clean_row.get(grade_column, "")
            json_file_name = clean_row.get(file_column, "")

            if grade in PASS_GRADES and json_file_name:
                selected_rows.append(
                    {
                        "pymol_grade": grade,
                        "json_file_name": json_file_name,
                    }
                )

    # ---------- 打印基本信息 ----------
    grade_counts = {
        grade: sum(row["pymol_grade"] == grade for row in selected_rows)
        for grade in PASS_GRADES
    }

    print("=" * 80)
    print("QUE candidate CIF extraction")
    print("=" * 80)
    print(f"输入 CSV 文件：{CSV_FILE}")
    print(f"原始 CIF 文件夹：{CIF_SOURCE_DIR}")
    print(f"候选输出文件夹：{CIF_OUTPUT_DIR}")
    print()
    print("通过筛选的 PyMOL 等级：")
    for grade in sorted(PASS_GRADES):
        print(f"  {grade}: {grade_counts[grade]}")
    print(f"合格候选总数：{len(selected_rows)}")
    print()

    # ---------- 复制对应 CIF 文件 ----------
    manifest_records = []
    copied_names = set()

    copied_count = 0
    missing_count = 0
    duplicate_count = 0

    for index, row in enumerate(selected_rows, start=1):
        grade = row["pymol_grade"]
        json_file_name = row["json_file_name"]

        # 转换规则：xxx.json -> xxx.cif
        cif_file_name = Path(json_file_name).with_suffix(".cif").name

        source_cif = CIF_SOURCE_DIR / cif_file_name

        # 输出到总文件夹
        output_cif = CIF_OUTPUT_DIR / cif_file_name

        # 同时输出到按等级分类的子文件夹
        grade_dir = get_grade_subfolder(grade)
        classified_output_cif = grade_dir / cif_file_name

        copy_status = ""

        if cif_file_name in copied_names:
            copy_status = "Duplicate_skipped"
            duplicate_count += 1
            print(f"[重复跳过] {cif_file_name}")

        elif not source_cif.exists():
            copy_status = "Missing_source_CIF"
            missing_count += 1
            print(f"[缺失] 未找到对应 CIF 文件：{cif_file_name}")

        else:
            # 复制到全部候选文件夹
            shutil.copy2(source_cif, output_cif)

            # 复制到对应等级子文件夹
            shutil.copy2(source_cif, classified_output_cif)

            copied_names.add(cif_file_name)
            copied_count += 1
            copy_status = "Copied"

            print(
                f"[{index:>3}/{len(selected_rows)}] 已复制："
                f"{cif_file_name} | {grade}"
            )

        manifest_records.append(
            {
                "selection_rank": index,
                "pymol_grade": grade,
                "json_file_name": json_file_name,
                "cif_file_name": cif_file_name,
                "source_cif": str(source_cif),
                "output_cif": str(output_cif),
                "classified_output_cif": str(classified_output_cif),
                "copy_status": copy_status,
            }
        )

    # ---------- 保存候选骨架清单 ----------
    manifest_fields = [
        "selection_rank",
        "pymol_grade",
        "json_file_name",
        "cif_file_name",
        "source_cif",
        "output_cif",
        "classified_output_cif",
        "copy_status",
    ]

    with MANIFEST_FILE.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_records)

    # ---------- 最终结果汇总 ----------
    print()
    print("=" * 80)
    print("提取完成")
    print("=" * 80)
    print(f"Excellent_QUE_buried_pocket 数量：{grade_counts['Excellent_QUE_buried_pocket']}")
    print(f"Good_QUE_buried_pocket 数量：{grade_counts['Good_QUE_buried_pocket']}")
    print(f"Good_QUE_partial_pocket 数量：{grade_counts['Good_QUE_partial_pocket']}")
    print(f"合格候选总数：{len(selected_rows)}")
    print(f"成功复制 CIF 文件数：{copied_count}")
    print(f"重复跳过数量：{duplicate_count}")
    print(f"未找到 CIF 文件数：{missing_count}")
    print()
    print(f"全部候选 CIF 文件夹：{CIF_OUTPUT_DIR}")
    print(f"候选骨架提取记录表：{MANIFEST_FILE}")
    print()
    print("分类子文件夹：")
    print(f"  {EXCELLENT_BURIED_DIR}")
    print(f"  {GOOD_BURIED_DIR}")
    print(f"  {GOOD_PARTIAL_DIR}")

    if missing_count > 0:
        print()
        print("以下 CIF 文件未找到，请检查文件名或原始 CIF 文件夹：")
        for record in manifest_records:
            if record["copy_status"] == "Missing_source_CIF":
                print(f"  - {record['cif_file_name']}")

    if copied_count == len(selected_rows) and missing_count == 0:
        print()
        print("结果检查通过：所有合格 QUE 候选骨架均已成功提取。")


if __name__ == "__main__":
    try:
        extract_que_candidate_cif()
    except Exception as exc:
        print("\n程序运行失败：")
        print(exc)
        sys.exit(1)