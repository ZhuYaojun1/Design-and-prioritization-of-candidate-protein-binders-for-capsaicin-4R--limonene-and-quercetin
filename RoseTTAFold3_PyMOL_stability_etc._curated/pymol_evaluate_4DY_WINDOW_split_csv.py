# -*- coding: utf-8 -*-
"""
pymol_evaluate_4DY_WINDOW_split_csv.py (修改版)
- 删除了部分输出列
- view_4DY_WINDOW 使用 super 叠合
"""

from pathlib import Path
from pymol import cmd
import csv
import math
import re
import numpy as np

# =========================================================
# 用户设置
# =========================================================

REFERENCE_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\4_RoseTTAFold3\4DY\Skeletal_structure"
)

PREDICTION_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\4_RoseTTAFold3\4DY\out_4DY_WINDOW"
)

OUTPUT_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\4_RoseTTAFold3\4DY\metrics_by_backbone"
)

WARNING_LOG = OUTPUT_DIR / "4DY_WINDOW_structure_metrics_warning_log.txt"

REFERENCE_PROTEIN_CHAIN = "A"
REFERENCE_LIGAND_CHAIN = "B"
PREDICTED_PROTEIN_CHAIN = "A"

SAMPLE_START = 1
SAMPLE_END = 10

POCKET_CUTOFF = 6.0
CONTACT_CUTOFF = 4.5
CLASH_CUTOFF = 2.0
HYDROPHOBIC_C_CUTOFF = 4.0

# 31个骨架列表 (保持不变)
BACKBONE_IDS = [
    "partial_0_model_0",
    "partial_0_model_1",
    "partial_0_model_7",
    "partial_0_model_9",
    "partial_1_model_2",
    "partial_1_model_7",
    "partial_1_model_8",
    "partial_2_model_3",
    "partial_2_model_4",
    "partial_2_model_6",
    "partial_3_model_1",
    "partial_3_model_7",
    "partial_3_model_8",
    "partial_4_model_0",
    "partial_4_model_6",
    "partial_4_model_7",
    "partial_5_model_1",
    "partial_5_model_3",
    "partial_5_model_4",
    "partial_5_model_5",
    "partial_5_model_6",
    "partial_6_model_2",
    "partial_7_model_0",
    "partial_7_model_4",
    "partial_7_model_5",
    "partial_7_model_7",
    "partial_7_model_8",
    "partial_8_model_2",
    "partial_8_model_3",
    "partial_8_model_6",
    "partial_9_model_5",
]

# 保留的输出字段（已删除指定的列）
FIELDS = [
    "backbone_id",
    "sample_id",
    "matched_CA_atom_count",
    "global_ca_rmsd_A",
    "global_backbone_rmsd_after_CA_fit_A",
    "pocket_ca_rmsd_after_CA_fit_A",
    "pocket_backbone_rmsd_after_CA_fit_A",
    "ref_ca_rg_A",
    "pred_ca_rg_A",
    "delta_ca_rg_A",
    "projected_min_protein_ligand_distance_A",
    "projected_delta_ligand_clash_pair_count",
    "projected_ligand_clash_pair_count_backbone",
    "projected_ligand_clash_pair_count_sidechain",
    "reference_contact_position_count",
    "projected_contact_position_count",
    "projected_contact_recovery_percent",
    "projected_new_contact_percent",
    "reference_hydrophobic_C_contact_pair_count",
    "projected_hydrophobic_C_contact_pair_count",
    "pred_mean_CA_Bfactor",
    "pred_pocket_mean_CA_Bfactor",
]

_REFERENCE_CACHE = {}


# =========================================================
# 通用计算函数 (与原脚本相同)
# =========================================================

def atom_records(selection):
    records = []
    for atom in cmd.get_model(selection, 1).atom:
        element = str(getattr(atom, "symbol", "") or getattr(atom, "elem", "")).upper()
        records.append({
            "chain": str(atom.chain).strip(),
            "resi": str(atom.resi).strip(),
            "resn": str(atom.resn).strip(),
            "name": str(atom.name).strip().upper(),
            "element": element,
            "coord": np.asarray(atom.coord, dtype=float),
            "b": float(getattr(atom, "b", float("nan"))),
        })
    return records


def heavy_atoms(selection):
    return [atom for atom in atom_records(selection) if atom["element"] != "H"]


def residue_sort_key(resi):
    match = re.search(r"-?\d+", str(resi))
    number = int(match.group()) if match else 10**9
    return (number, str(resi))


def atom_coordinate_dict(selection, allowed_names):
    allowed_names = set(allowed_names)
    output = {}
    for atom in atom_records(selection):
        if atom["name"] in allowed_names:
            output[(atom["resi"], atom["name"])] = atom["coord"]
    return output


def matched_keys(reference_dict, prediction_dict, residue_subset=None):
    keys = set(reference_dict).intersection(prediction_dict)
    if residue_subset is not None:
        keys = {key for key in keys if key[0] in residue_subset}
    return sorted(keys, key=lambda key: (residue_sort_key(key[0]), key[1]))


def coordinates(mapping, keys):
    return np.asarray([mapping[key] for key in keys], dtype=float)


def kabsch_transform(mobile, target):
    if len(mobile) < 3 or len(mobile) != len(target):
        raise ValueError("匹配的 Cα 原子不足 3 对，无法建立结构叠合。")

    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    mobile_zero = mobile - mobile_center
    target_zero = target - target_center

    u, _, vt = np.linalg.svd(mobile_zero.T @ target_zero)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0

    rotation = u @ correction @ vt
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def transform_coords(coords, rotation, translation):
    return np.asarray(coords, dtype=float) @ rotation + translation


def rmsd(coords_a, coords_b):
    if len(coords_a) == 0 or len(coords_a) != len(coords_b):
        return float("nan")
    delta = np.asarray(coords_a) - np.asarray(coords_b)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def radius_of_gyration(coords):
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return float("nan")
    center = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


def distance_matrix(coords_a, coords_b):
    coords_a = np.asarray(coords_a, dtype=float)
    coords_b = np.asarray(coords_b, dtype=float)
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.empty((0, 0), dtype=float)
    return np.sqrt(np.sum((coords_a[:, None, :] - coords_b[None, :, :]) ** 2, axis=2))


def contacting_positions(protein_atoms, ligand_atoms, cutoff):
    protein_xyz = [atom["coord"] for atom in protein_atoms]
    ligand_xyz = [atom["coord"] for atom in ligand_atoms]
    distances = distance_matrix(protein_xyz, ligand_xyz)
    if distances.size == 0:
        return set()

    atom_indexes = np.where(np.any(distances <= cutoff, axis=1))[0]
    return {protein_atoms[index]["resi"] for index in atom_indexes}


def transform_atom_records(atoms, rotation, translation):
    transformed = []
    for atom in atoms:
        new_atom = dict(atom)
        new_atom["coord"] = transform_coords([atom["coord"]], rotation, translation)[0]
        transformed.append(new_atom)
    return transformed


def clash_count(protein_atoms, ligand_atoms):
    distances = distance_matrix(
        [atom["coord"] for atom in protein_atoms],
        [atom["coord"] for atom in ligand_atoms],
    )
    return int(np.sum(distances < CLASH_CUTOFF)) if distances.size else 0


def min_distance(protein_atoms, ligand_atoms):
    distances = distance_matrix(
        [atom["coord"] for atom in protein_atoms],
        [atom["coord"] for atom in ligand_atoms],
    )
    return float(np.min(distances)) if distances.size else float("nan")


def hydrophobic_carbon_contacts(protein_atoms, ligand_atoms):
    protein_carbon = [atom["coord"] for atom in protein_atoms if atom["element"] == "C"]
    ligand_carbon = [atom["coord"] for atom in ligand_atoms if atom["element"] == "C"]
    distances = distance_matrix(protein_carbon, ligand_carbon)
    return int(np.sum(distances <= HYDROPHOBIC_C_CUTOFF)) if distances.size else 0


def mean_bfactor(selection):
    b_values = [atom["b"] for atom in atom_records(selection) if not math.isnan(atom["b"])]
    return float(np.mean(b_values)) if b_values else float("nan")


def rounded(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return round(float(value), 4)


# =========================================================
# 文件解析函数
# =========================================================

def reference_path_for(backbone_id):
    return REFERENCE_DIR / f"{backbone_id}.cif"


def find_predicted_path(backbone_id, sample_id):
    """
    兼容以下常见 RF3 输出布局：
    1) out_4DY_WINDOW/partial_x_model_y_sampleN_model.cif
    2) out_4DY_WINDOW/partial_x_model_y_sampleN/partial_x_model_y_sampleN_model.cif
    3) out_4DY_WINDOW 下面任意一层中的同名 CIF 文件
    """
    expected_name = f"{backbone_id}_sample{sample_id}_model.cif"

    direct = PREDICTION_DIR / expected_name
    if direct.exists():
        return direct

    nested = PREDICTION_DIR / f"{backbone_id}_sample{sample_id}" / expected_name
    if nested.exists():
        return nested

    candidates = list(PREDICTION_DIR.rglob(expected_name))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"找到多个匹配的预测结构：{expected_name}")
    return None


# =========================================================
# 读取 reference
# =========================================================

def read_reference(backbone_id):
    if backbone_id in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[backbone_id]

    reference_path = reference_path_for(backbone_id)
    if not reference_path.exists():
        raise FileNotFoundError(f"找不到参考骨架：{reference_path}")

    cmd.delete("__reference")
    cmd.load(str(reference_path), "__reference")

    protein_selection = f"__reference and chain {REFERENCE_PROTEIN_CHAIN}"
    ligand_selection = f"__reference and chain {REFERENCE_LIGAND_CHAIN}"

    ref_protein_heavy = heavy_atoms(protein_selection)
    ref_ligand_heavy = heavy_atoms(ligand_selection)

    if not ref_protein_heavy:
        raise ValueError(f"{reference_path.name} 中找不到蛋白链 {REFERENCE_PROTEIN_CHAIN}。")
    if not ref_ligand_heavy:
        raise ValueError(f"{reference_path.name} 中找不到小分子链 {REFERENCE_LIGAND_CHAIN}。")

    ref_ca = atom_coordinate_dict(protein_selection + " and name CA", {"CA"})
    ref_backbone = atom_coordinate_dict(
        protein_selection + " and name N+CA+C+O", {"N", "CA", "C", "O"}
    )

    pocket_positions = contacting_positions(ref_protein_heavy, ref_ligand_heavy, POCKET_CUTOFF)
    contact_positions = contacting_positions(ref_protein_heavy, ref_ligand_heavy, CONTACT_CUTOFF)

    reference_data = {
        "path": reference_path,
        "protein_heavy": ref_protein_heavy,
        "ligand_heavy": ref_ligand_heavy,
        "ca": ref_ca,
        "backbone": ref_backbone,
        "pocket_positions": pocket_positions,
        "contact_positions": contact_positions,
        "ligand_resn": ";".join(sorted({atom["resn"] for atom in ref_ligand_heavy})),
        "ligand_heavy_atom_count": len(ref_ligand_heavy),
        "ca_rg": radius_of_gyration(list(ref_ca.values())),
        "clash_count": clash_count(ref_protein_heavy, ref_ligand_heavy),
        "hydrophobic_contacts": hydrophobic_carbon_contacts(ref_protein_heavy, ref_ligand_heavy),
    }

    cmd.delete("__reference")
    _REFERENCE_CACHE[backbone_id] = reference_data
    return reference_data


# =========================================================
# 评价单个 RF3 结构
# =========================================================

def evaluate_one(backbone_id, sample_id):
    # 只保留需要输出的字段
    result = {field: "" for field in FIELDS}
    result["backbone_id"] = backbone_id
    result["sample_id"] = sample_id

    reference = read_reference(backbone_id)
    predicted_path = find_predicted_path(backbone_id, sample_id)

    # 内部仍需使用这些值，但不写入 CSV
    # result["reference_file"] = ...    # 已删除
    # result["reference_ligand_resn"] = ... # 已删除
    # 等等

    if predicted_path is None:
        raise FileNotFoundError(f"找不到 {backbone_id} sample{sample_id} 的 RF3 CIF 文件。")

    # 仍需要加载预测文件进行叠合计算
    cmd.delete("__predicted")
    cmd.load(str(predicted_path), "__predicted")

    pred_selection = f"__predicted and chain {PREDICTED_PROTEIN_CHAIN}"
    pred_protein_heavy = heavy_atoms(pred_selection)
    if not pred_protein_heavy:
        raise ValueError(f"{predicted_path.name} 中找不到预测蛋白链 {PREDICTED_PROTEIN_CHAIN}。")

    pred_ca = atom_coordinate_dict(pred_selection + " and name CA", {"CA"})
    pred_backbone = atom_coordinate_dict(
        pred_selection + " and name N+CA+C+O", {"N", "CA", "C", "O"}
    )

    # 叠合计算（与原来完全相同）
    ca_keys = matched_keys(reference["ca"], pred_ca)
    ref_ca_coords = coordinates(reference["ca"], ca_keys)
    pred_ca_coords = coordinates(pred_ca, ca_keys)
    rotation, translation = kabsch_transform(pred_ca_coords, ref_ca_coords)

    result["matched_CA_atom_count"] = len(ca_keys)
    result["global_ca_rmsd_A"] = rounded(
        rmsd(transform_coords(pred_ca_coords, rotation, translation), ref_ca_coords)
    )
    result["ref_ca_rg_A"] = rounded(reference["ca_rg"])
    result["pred_ca_rg_A"] = rounded(radius_of_gyration(pred_ca_coords))
    result["delta_ca_rg_A"] = rounded(radius_of_gyration(pred_ca_coords) - reference["ca_rg"])

    bb_keys = matched_keys(reference["backbone"], pred_backbone)
    ref_bb_coords = coordinates(reference["backbone"], bb_keys)
    pred_bb_coords = coordinates(pred_backbone, bb_keys)
    # matched_backbone_atom_count 不再输出，但仍可用于调试
    # result["matched_backbone_atom_count"] = len(bb_keys)
    result["global_backbone_rmsd_after_CA_fit_A"] = rounded(
        rmsd(transform_coords(pred_bb_coords, rotation, translation), ref_bb_coords)
    )

    pocket_ca_keys = matched_keys(reference["ca"], pred_ca, reference["pocket_positions"])
    if pocket_ca_keys:
        ref_pocket_ca = coordinates(reference["ca"], pocket_ca_keys)
        pred_pocket_ca = coordinates(pred_ca, pocket_ca_keys)
        result["pocket_ca_rmsd_after_CA_fit_A"] = rounded(
            rmsd(transform_coords(pred_pocket_ca, rotation, translation), ref_pocket_ca)
        )

    pocket_bb_keys = matched_keys(reference["backbone"], pred_backbone, reference["pocket_positions"])
    if pocket_bb_keys:
        ref_pocket_bb = coordinates(reference["backbone"], pocket_bb_keys)
        pred_pocket_bb = coordinates(pred_backbone, pocket_bb_keys)
        result["pocket_backbone_rmsd_after_CA_fit_A"] = rounded(
            rmsd(transform_coords(pred_pocket_bb, rotation, translation), ref_pocket_bb)
        )

    # 将预测蛋白变换到参考坐标系
    aligned_pred_heavy = transform_atom_records(pred_protein_heavy, rotation, translation)
    aligned_pred_backbone = [
        atom for atom in aligned_pred_heavy if atom["name"] in {"N", "CA", "C", "O"}
    ]
    aligned_pred_sidechain = [
        atom for atom in aligned_pred_heavy if atom["name"] not in {"N", "CA", "C", "O"}
    ]

    result["projected_min_protein_ligand_distance_A"] = rounded(
        min_distance(aligned_pred_heavy, reference["ligand_heavy"])
    )

    predicted_clash = clash_count(aligned_pred_heavy, reference["ligand_heavy"])
    # 不再输出 projected_ligand_clash_pair_count_all_heavy 和 reference_ligand_clash_pair_count_all_heavy
    result["projected_delta_ligand_clash_pair_count"] = predicted_clash - reference["clash_count"]
    result["projected_ligand_clash_pair_count_backbone"] = clash_count(
        aligned_pred_backbone, reference["ligand_heavy"]
    )
    result["projected_ligand_clash_pair_count_sidechain"] = clash_count(
        aligned_pred_sidechain, reference["ligand_heavy"]
    )

    # 接触相关
    projected_contacts = contacting_positions(
        aligned_pred_heavy, reference["ligand_heavy"], CONTACT_CUTOFF
    )
    recovered_contacts = projected_contacts.intersection(reference["contact_positions"])
    new_contacts = projected_contacts.difference(reference["contact_positions"])

    result["reference_contact_position_count"] = len(reference["contact_positions"])
    result["projected_contact_position_count"] = len(projected_contacts)
    if reference["contact_positions"]:
        result["projected_contact_recovery_percent"] = rounded(
            100.0 * len(recovered_contacts) / len(reference["contact_positions"])
        )
    if projected_contacts:
        result["projected_new_contact_percent"] = rounded(
            100.0 * len(new_contacts) / len(projected_contacts)
        )

    result["reference_hydrophobic_C_contact_pair_count"] = reference["hydrophobic_contacts"]
    result["projected_hydrophobic_C_contact_pair_count"] = hydrophobic_carbon_contacts(
        aligned_pred_heavy, reference["ligand_heavy"]
    )

    result["pred_mean_CA_Bfactor"] = rounded(mean_bfactor(pred_selection + " and name CA"))

    pocket_b_values = [
        atom["b"] for atom in atom_records(pred_selection + " and name CA")
        if atom["resi"] in reference["pocket_positions"]
    ]
    if pocket_b_values:
        result["pred_pocket_mean_CA_Bfactor"] = rounded(float(np.mean(pocket_b_values)))

    cmd.delete("__predicted")
    return result


# =========================================================
# 批量运行：31 个骨架分别输出 CSV
# =========================================================

def evaluate_4DY_WINDOW():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _REFERENCE_CACHE.clear()

    warnings = []
    total_rows = 0
    successful_rows = 0
    output_csv_count = 0

    print("====================================================")
    print("开始进行 4DY_WINDOW RF3 结构评价")
    print("====================================================")
    print(f"Reference 骨架数：{len(BACKBONE_IDS)}")
    print(f"每个骨架预计 RF3 结构数：{SAMPLE_END - SAMPLE_START + 1}")
    print(f"预计评价结构总数：{len(BACKBONE_IDS) * (SAMPLE_END - SAMPLE_START + 1)}")
    print(f"参考骨架目录：{REFERENCE_DIR}")
    print(f"RF3 预测目录：{PREDICTION_DIR}")
    print(f"输出目录：{OUTPUT_DIR}")

    for backbone_id in BACKBONE_IDS:
        reference_path = reference_path_for(backbone_id)

        if not reference_path.exists():
            warning = f"[缺失参考骨架] {reference_path}"
            warnings.append(warning)
            print(warning)
            continue

        rows = []
        backbone_success = 0

        for sample_id in range(SAMPLE_START, SAMPLE_END + 1):
            try:
                result = evaluate_one(backbone_id, sample_id)
                rows.append(result)
                backbone_success += 1
                successful_rows += 1
            except Exception as error:
                # 失败时输出空的占位行，但只包含 backbone_id 和 sample_id，其他字段留空
                failed = {field: "" for field in FIELDS}
                failed["backbone_id"] = backbone_id
                failed["sample_id"] = sample_id
                rows.append(failed)
                warning = f"[评价失败] {backbone_id} sample{sample_id}: {error}"
                warnings.append(warning)
                # 注意：不再写入 status 和 error 列，所以 failed 字典中不需要这些键

        output_csv = OUTPUT_DIR / f"{backbone_id}_RF3_metrics.csv"
        with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        output_csv_count += 1
        total_rows += len(rows)
        print(f"[完成] {backbone_id}: 成功评价 {backbone_success}/10；输出 {output_csv.name}")

    if warnings:
        WARNING_LOG.write_text("\n".join(warnings), encoding="utf-8")
    else:
        WARNING_LOG.write_text("无警告：31 个骨架及对应 RF3 结构均成功评价。\n", encoding="utf-8")

    print("\n====================================================")
    print("4DY_WINDOW 结构评价完成")
    print("====================================================")
    print(f"生成 CSV 数量：{output_csv_count}（预期 31）")
    print(f"总结果行数：{total_rows}（预期 310）")
    print(f"成功评价结构数：{successful_rows}")
    print(f"失败结构数：{total_rows - successful_rows}")
    print(f"结果目录：{OUTPUT_DIR}")
    print(f"日志文件：{WARNING_LOG}")
    print("\n注意：projected_ligand_* 指标评价的是 RF3 回折蛋白")
    print("对 RFD3 原始配体位置的几何兼容性，不是 RF3 预测出的配体姿态。")


# =========================================================
# 可视化单个配对（使用 super 叠合）
# =========================================================

def view_4DY_WINDOW(backbone_id, sample_id):
    sample_id = int(sample_id)
    reference_path = reference_path_for(backbone_id)
    predicted_path = find_predicted_path(backbone_id, sample_id)

    if not reference_path.exists():
        print(f"找不到 reference：{reference_path}")
        return
    if predicted_path is None:
        print(f"找不到 predicted structure：{backbone_id} sample{sample_id}")
        return

    cmd.delete("ref_view")
    cmd.delete("rf3_view")
    cmd.load(str(reference_path), "ref_view")
    cmd.load(str(predicted_path), "rf3_view")

    # 使用 super 命令进行结构叠合（基于序列和结构比对）
    cmd.super("rf3_view and chain A", "ref_view and chain A", object="super_aligned")

    cmd.hide("everything", "all")
    cmd.show("cartoon", "ref_view and chain A")
    cmd.show("cartoon", "rf3_view and chain A")
    cmd.show("sticks", "ref_view and chain B")
    cmd.color("cyan", "ref_view and chain A")
    cmd.color("palegreen", "rf3_view and chain A")
    cmd.color("yellow", "ref_view and chain B")
    cmd.zoom("ref_view and chain B", 8)

    print(f"已显示 {backbone_id} sample{sample_id}：cyan=RFD3蛋白；palegreen=RF3蛋白；yellow=RFD3配体（super叠合）。")


cmd.extend("evaluate_4DY_WINDOW", evaluate_4DY_WINDOW)
cmd.extend("view_4DY_WINDOW", view_4DY_WINDOW)

print("脚本加载成功。")
print("批量计算命令：evaluate_4DY_WINDOW")
print("单个结构查看命令：view_4DY_WINDOW partial_0_model_0, 1")