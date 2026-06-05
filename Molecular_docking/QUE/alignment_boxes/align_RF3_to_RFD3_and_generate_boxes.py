from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, Select, Superimposer
from Bio.PDB.Polypeptide import is_aa


# ============================================================
# 基础路径
# ============================================================
ROOT = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\6_AutoDock_Vina\QUE")

PAIRING_CSV = ROOT / "00_input_check" / "QUE_32_receptor_RFD3_pairing.csv"

WORK_DIR = ROOT / "01_alignment_boxes"
ALIGNED_DIR = WORK_DIR / "aligned_RF3_PDB"
LIGAND_REF_DIR = WORK_DIR / "reference_QUE_PDB"
CONFIG_DIR = WORK_DIR / "vina_box_configs"

SUMMARY_CSV = WORK_DIR / "QUE_32_alignment_box_summary.csv"
ERROR_CSV = WORK_DIR / "QUE_alignment_box_errors.csv"
SUMMARY_TXT = WORK_DIR / "QUE_alignment_box_summary.txt"

QUE_LIGAND_PDBQT = ROOT / "QUE_PDBQT" / "QUE_ideal.pdbqt"
FUTURE_RECEPTOR_PDBQT_DIR = ROOT / "02_prepared_PDBQT" / "receptors"
FUTURE_DOCKING_OUTPUT_DIR = ROOT / "03_vina_docking" / "outputs"


# ============================================================
# 参数
# ============================================================
# RFD3 复合物中配体的优先残基名称。
# 如果实际名称不是 QUE，脚本还会自动搜索非蛋白、非水分子。
PREFERRED_LIGAND_NAMES = {"QUE"}

# 每个方向的搜索盒最小边长，单位 Å。
# 对 QUE 这样的中等大小小分子，20 Å 可用于局部口袋 redocking 初筛。
MIN_BOX_SIZE = 20.0

# 在配体实际空间跨度基础上增加的额外余量，单位 Å。
BOX_PADDING = 8.0

# 若 box 某一边超过该值则给出警告。
BOX_WARNING_LIMIT = 30.0

# Vina 初筛参数。
EXHAUSTIVENESS = 32
NUM_MODES = 10
ENERGY_RANGE = 5


# ============================================================
# 结构读取函数
# ============================================================
def load_structure(path: Path, structure_id: str):
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    elif suffix in {".pdb", ".ent"}:
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"暂不支持用于对齐的结构格式：{path.name}")
    return parser.get_structure(structure_id, str(path))


def first_model(structure):
    return next(structure.get_models())


def collect_protein_ca_atoms(structure) -> List:
    """
    提取蛋白部分的 Cα 原子。
    优先识别标准/非标准氨基酸；若失败，则回退至所有含 CA 且不是 QUE/水的残基。
    """
    model = first_model(structure)
    ca_atoms = []

    for chain in model:
        for residue in chain:
            if is_aa(residue, standard=False) and "CA" in residue:
                ca_atoms.append(residue["CA"])

    if len(ca_atoms) >= 5:
        return ca_atoms

    ca_atoms = []
    for chain in model:
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            if resname not in PREFERRED_LIGAND_NAMES and resname not in {"HOH", "WAT"} and "CA" in residue:
                ca_atoms.append(residue["CA"])

    return ca_atoms


def atom_is_hydrogen(atom) -> bool:
    element = (getattr(atom, "element", "") or "").strip().upper()
    if element:
        return element == "H"
    return atom.get_name().strip().upper().startswith("H")


def collect_ligand_residues(structure):
    """
    在 RFD3 复合物中寻找 QUE。
    优先使用残基名 QUE。
    若未找到，则寻找非蛋白、非水、非常见无机离子的异源分子。
    """
    model = first_model(structure)

    preferred = []
    for chain in model:
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            if resname in PREFERRED_LIGAND_NAMES:
                preferred.append(residue)

    if preferred:
        return preferred, "preferred_resname"

    excluded = {
        "HOH", "WAT", "NA", "CL", "K", "CA", "MG", "ZN", "MN",
        "SO4", "PO4", "PEG", "GOL", "EDO"
    }

    hetero_candidates = []
    for chain in model:
        for residue in chain:
            resname = residue.get_resname().strip().upper()
            hetflag = str(residue.id[0]).strip()
            if (
                not is_aa(residue, standard=False)
                and resname not in excluded
                and hetflag != ""
            ):
                heavy_atoms = [a for a in residue.get_atoms() if not atom_is_hydrogen(a)]
                if len(heavy_atoms) >= 5:
                    hetero_candidates.append(residue)

    if len(hetero_candidates) == 1:
        return hetero_candidates, "single_hetero_candidate"

    candidate_names = sorted({
        residue.get_resname().strip().upper() for residue in hetero_candidates
    })
    raise ValueError(
        "未能唯一识别 QUE 配体。"
        f"检测到的非蛋白候选残基名称为：{candidate_names}。"
        "请确认 RFD3 复合物中的配体残基名。"
    )


def ligand_heavy_atom_coordinates(ligand_residues) -> np.ndarray:
    coords = []
    for residue in ligand_residues:
        for atom in residue.get_atoms():
            if not atom_is_hydrogen(atom):
                coords.append(atom.coord)

    if not coords:
        raise ValueError("QUE 未检测到重原子，无法计算 docking box。")

    return np.asarray(coords, dtype=float)


def write_ligand_pdb(ligand_residues, output_path: Path):
    """
    写出用于后续姿势恢复评价的设计态 QUE 参考 PDB。
    """
    atom_serial = 1
    lines = []

    for residue in ligand_residues:
        resname = residue.get_resname().strip() or "QUE"
        chain_id = residue.get_parent().id.strip() or "L"
        resid = residue.id[1] if isinstance(residue.id[1], int) else 1

        for atom in residue.get_atoms():
            x, y, z = atom.coord
            atom_name = atom.get_name().strip()
            element = (getattr(atom, "element", "") or atom_name[0]).strip().upper()

            line = (
                f"HETATM{atom_serial:5d} {atom_name:<4s} {resname:>3s} {chain_id:1s}"
                f"{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}\n"
            )
            lines.append(line)
            atom_serial += 1

    lines.append("END\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def calculate_rmsd_before_alignment(fixed_atoms, moving_atoms) -> float:
    fixed = np.asarray([a.coord for a in fixed_atoms], dtype=float)
    moving = np.asarray([a.coord for a in moving_atoms], dtype=float)

    diff = fixed - moving
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def calculate_box(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    extent = max_xyz - min_xyz
    center = coords.mean(axis=0)

    sizes = np.maximum(extent + BOX_PADDING, MIN_BOX_SIZE)
    return center, sizes, extent


class ProteinOnlySelect(Select):
    def accept_residue(self, residue):
        return 1 if is_aa(residue, standard=False) else 0


# ============================================================
# 主流程
# ============================================================
def main():
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    LIGAND_REF_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not PAIRING_CSV.exists():
        raise FileNotFoundError(f"未找到配对文件：{PAIRING_CSV}")

    with PAIRING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    results = []
    errors = []

    for row in rows:
        selected_name = row["Selected_Name"].strip()
        receptor_path = Path(row["Receptor_Path"])
        rfd3_path = Path(row["RFD3_Complex_Path"])

        print(f"[Processing] {selected_name}")

        try:
            if not receptor_path.exists():
                raise FileNotFoundError(f"RF3 receptor 不存在：{receptor_path}")

            if not rfd3_path.exists():
                raise FileNotFoundError(f"RFD3 complex 不存在：{rfd3_path}")

            rf3_structure = load_structure(receptor_path, f"{selected_name}_RF3")
            rfd3_structure = load_structure(rfd3_path, f"{selected_name}_RFD3")

            rf3_ca = collect_protein_ca_atoms(rf3_structure)
            rfd3_ca = collect_protein_ca_atoms(rfd3_structure)

            if len(rf3_ca) < 5 or len(rfd3_ca) < 5:
                raise ValueError(
                    f"蛋白 Cα 原子不足：RF3={len(rf3_ca)}, RFD3={len(rfd3_ca)}"
                )

            if len(rf3_ca) != len(rfd3_ca):
                raise ValueError(
                    f"RF3 与 RFD3 的 Cα 数量不一致：RF3={len(rf3_ca)}, "
                    f"RFD3={len(rfd3_ca)}。需要人工核查链或残基编号。"
                )

            rmsd_before = calculate_rmsd_before_alignment(rfd3_ca, rf3_ca)

            superimposer = Superimposer()
            superimposer.set_atoms(rfd3_ca, rf3_ca)
            superimposer.apply(list(rf3_structure.get_atoms()))

            rmsd_after = float(superimposer.rms)

            aligned_pdb = ALIGNED_DIR / f"{selected_name}_aligned_RF3.pdb"
            io = PDBIO()
            io.set_structure(rf3_structure)
            io.save(str(aligned_pdb), ProteinOnlySelect())

            ligand_residues, ligand_detection_method = collect_ligand_residues(rfd3_structure)
            ligand_coords = ligand_heavy_atom_coordinates(ligand_residues)

            ligand_resnames = ";".join(sorted({
                residue.get_resname().strip().upper() for residue in ligand_residues
            }))

            ligand_ref_pdb = LIGAND_REF_DIR / f"{selected_name}_reference_QUE.pdb"
            write_ligand_pdb(ligand_residues, ligand_ref_pdb)

            center, sizes, extent = calculate_box(ligand_coords)

            box_warning = ""
            if np.any(sizes > BOX_WARNING_LIMIT):
                box_warning = "box_dimension_above_30A"

            future_receptor_pdbqt = FUTURE_RECEPTOR_PDBQT_DIR / f"{selected_name}_aligned_RF3.pdbqt"
            future_output_pdbqt = FUTURE_DOCKING_OUTPUT_DIR / f"{selected_name}_QUE_out.pdbqt"

            vina_config = CONFIG_DIR / f"{selected_name}_vina_box.txt"
            config_text = (
                f"receptor = {future_receptor_pdbqt}\n"
                f"ligand = {QUE_LIGAND_PDBQT}\n"
                f"\n"
                f"center_x = {center[0]:.3f}\n"
                f"center_y = {center[1]:.3f}\n"
                f"center_z = {center[2]:.3f}\n"
                f"\n"
                f"size_x = {sizes[0]:.3f}\n"
                f"size_y = {sizes[1]:.3f}\n"
                f"size_z = {sizes[2]:.3f}\n"
                f"\n"
                f"exhaustiveness = {EXHAUSTIVENESS}\n"
                f"num_modes = {NUM_MODES}\n"
                f"energy_range = {ENERGY_RANGE}\n"
                f"\n"
                f"out = {future_output_pdbqt}\n"
            )
            vina_config.write_text(config_text, encoding="utf-8")

            results.append({
                "Selected_Name": selected_name,
                "RF3_Source_File": str(receptor_path),
                "RFD3_Complex_File": str(rfd3_path),
                "Aligned_RF3_PDB": str(aligned_pdb),
                "Reference_QUE_PDB": str(ligand_ref_pdb),
                "Ligand_Resname": ligand_resnames,
                "Ligand_Detection_Method": ligand_detection_method,
                "Protein_CA_Atom_Count": len(rf3_ca),
                "RMSD_Before_Alignment_A": round(rmsd_before, 4),
                "RMSD_After_Alignment_A": round(rmsd_after, 4),
                "QUE_Heavy_Atom_Count": len(ligand_coords),
                "QUE_Extent_X_A": round(float(extent[0]), 3),
                "QUE_Extent_Y_A": round(float(extent[1]), 3),
                "QUE_Extent_Z_A": round(float(extent[2]), 3),
                "Center_X": round(float(center[0]), 3),
                "Center_Y": round(float(center[1]), 3),
                "Center_Z": round(float(center[2]), 3),
                "Size_X": round(float(sizes[0]), 3),
                "Size_Y": round(float(sizes[1]), 3),
                "Size_Z": round(float(sizes[2]), 3),
                "Box_Warning": box_warning,
                "Vina_Config_File": str(vina_config),
                "Status": "Success"
            })

            print(
                f"  Success | CA={len(rf3_ca)} | RMSD_after={rmsd_after:.4f} Å | "
                f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) | "
                f"size=({sizes[0]:.3f}, {sizes[1]:.3f}, {sizes[2]:.3f})"
            )

        except Exception as exc:
            errors.append({
                "Selected_Name": selected_name,
                "RF3_Source_File": str(receptor_path),
                "RFD3_Complex_File": str(rfd3_path),
                "Error": str(exc)
            })
            print(f"  ERROR | {exc}")

    if results:
        with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    if errors:
        with ERROR_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(errors[0].keys()))
            writer.writeheader()
            writer.writerows(errors)

    success_count = len(results)
    error_count = len(errors)

    if results:
        rmsd_values = np.array([r["RMSD_After_Alignment_A"] for r in results], dtype=float)
        median_rmsd = float(np.median(rmsd_values))
        max_rmsd = float(np.max(rmsd_values))
        high_rmsd_count = int(np.sum(rmsd_values > 1.0))
    else:
        median_rmsd = float("nan")
        max_rmsd = float("nan")
        high_rmsd_count = 0

    summary_lines = [
        "QUE RF3–RFD3 对齐与 docking box 生成摘要",
        "=" * 66,
        f"输入配对结构数量：{len(rows)}",
        f"成功完成对齐并生成 box 数量：{success_count}",
        f"失败数量：{error_count}",
        "",
        "对齐质量：",
        f"对齐后 Cα RMSD 中位数：{median_rmsd:.4f} Å" if results else "对齐后 Cα RMSD 中位数：NA",
        f"对齐后 Cα RMSD 最大值：{max_rmsd:.4f} Å" if results else "对齐后 Cα RMSD 最大值：NA",
        f"对齐后 Cα RMSD > 1.0 Å 的结构数量：{high_rmsd_count}",
        "",
        "输出文件：",
        f"对齐后的 RF3 PDB 文件夹：{ALIGNED_DIR}",
        f"设计态 QUE 参考结构文件夹：{LIGAND_REF_DIR}",
        f"Vina box 配置文件夹：{CONFIG_DIR}",
        f"汇总结果 CSV：{SUMMARY_CSV}",
        f"错误记录 CSV：{ERROR_CSV if errors else '无'}",
        "=" * 66
    ]

    SUMMARY_TXT.write_text("\n".join(summary_lines), encoding="utf-8")

    print("")
    print("=" * 66)
    print("QUE RF3–RFD3 对齐与 docking box 生成完成")
    print("=" * 66)
    print(f"输入配对结构数量：{len(rows)}")
    print(f"成功数量：{success_count}")
    print(f"失败数量：{error_count}")

    if results:
        print(f"对齐后 Cα RMSD 中位数：{median_rmsd:.4f} Å")
        print(f"对齐后 Cα RMSD 最大值：{max_rmsd:.4f} Å")
        print(f"对齐后 Cα RMSD > 1.0 Å 数量：{high_rmsd_count}")

    print("")
    print(f"结果汇总表：{SUMMARY_CSV}")
    print(f"结果摘要：{SUMMARY_TXT}")

    if errors:
        print(f"错误记录：{ERROR_CSV}")
        sys.exit(1)


if __name__ == "__main__":
    main()
