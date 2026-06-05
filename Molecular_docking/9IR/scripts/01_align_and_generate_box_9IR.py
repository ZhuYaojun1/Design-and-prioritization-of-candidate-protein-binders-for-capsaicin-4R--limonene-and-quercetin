from pathlib import Path
import csv
import re
from pymol import cmd

# ============================================================
# 基本路径
# ============================================================

ROOT = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\6_AutoDock_Vina\9IR")

RECEPTOR_DIR = ROOT / "Receptor_PDBQT"       # RF3 预测纯蛋白 CIF
REFERENCE_DIR = ROOT / "RFD3_cifs"           # RFD3 蛋白-9IR 参考复合物 CIF

ALIGNED_DIR = ROOT / "Aligned_Receptor_PDB"
CONFIG_DIR = ROOT / "box_parameters" / "configs"
REFERENCE_LIGAND_DIR = ROOT / "box_parameters" / "reference_ligands"
QC_DIR = ROOT / "QC"

for folder in [ALIGNED_DIR, CONFIG_DIR, REFERENCE_LIGAND_DIR, QC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 9IR 为较小的 buried hydrophobic ligand。
# 第一轮 redocking 采用统一最小 20 Å 立方体搜索框。
MIN_BOX_SIZE = 20.0

results = []

# ============================================================
# 逐个处理 54 个 receptor
# ============================================================

receptor_files = sorted(RECEPTOR_DIR.glob("*_sample*_model.cif"))

print(f"Detected receptor files: {len(receptor_files)}")

for receptor_file in receptor_files:

    receptor_id = receptor_file.stem

    # 例如：
    # buried_0_model_0_sample1_model -> buried_0_model_0
    backbone_id = re.sub(r"_sample\d+_model$", "", receptor_id)

    reference_file = REFERENCE_DIR / f"{backbone_id}.cif"

    if not reference_file.exists():
        results.append({
            "receptor_id": receptor_id,
            "backbone_id": backbone_id,
            "status": "missing_reference"
        })
        print(f"[FAILED] Missing reference: {receptor_id}")
        continue

    cmd.reinitialize()

    cmd.load(str(reference_file), "ref")
    cmd.load(str(receptor_file), "rec")

    # ========================================================
    # 识别参考复合物中的 9IR
    # ========================================================

    ligand_selection = "ref and resn 9IR"

    ligand_atom_count = cmd.count_atoms(ligand_selection)

    if ligand_atom_count == 0:
        results.append({
            "receptor_id": receptor_id,
            "backbone_id": backbone_id,
            "reference_file": str(reference_file),
            "status": "ligand_not_found"
        })
        print(f"[FAILED] 9IR not found: {receptor_id}")
        continue

    # ========================================================
    # 提取 RFD3 中 9IR 周围 5 Å 内的口袋 Cα 原子
    # ========================================================

    # 先识别任意重原子与 9IR 距离小于 5 Å 的口袋残基，
    # 再提取这些残基的 Cα 进行局部口袋拟合。
    # 这对于 9IR 的 buried hydrophobic pocket 尤其重要，
    # 因为真实接触通常来自侧链而不是 Cα 本身。
    pocket_ca_selection = (
        f"(byres ((ref and polymer.protein) within 5.0 of ({ligand_selection}))) "
        f"and ref and polymer.protein and name CA"
    )

    pocket_atoms = cmd.get_model(pocket_ca_selection).atom

    fit_pairs = []

    for atom in pocket_atoms:

        # RFD3 reference 中的目标原子
        target_sel = (
            f"ref and polymer.protein and chain {atom.chain} "
            f"and resi {atom.resi} and name CA"
        )

        # 优先尝试按照相同 chain + resi 在 RF3 receptor 中寻找
        mobile_sel = (
            f"rec and polymer.protein and chain {atom.chain} "
            f"and resi {atom.resi} and name CA"
        )

        # 如果 RF3 的链名称发生改变，则只按照残基编号匹配
        if cmd.count_atoms(mobile_sel) != 1:
            mobile_sel = (
                f"rec and polymer.protein and resi {atom.resi} and name CA"
            )

        if (
            cmd.count_atoms(target_sel) == 1
            and cmd.count_atoms(mobile_sel) == 1
        ):
            fit_pairs.extend([mobile_sel, target_sel])

    pocket_ca_count = len(fit_pairs) // 2

    # ========================================================
    # 将 RF3 receptor 对齐到 RFD3 参考口袋坐标系
    # ========================================================

    if pocket_ca_count >= 3:

        alignment_method = "pocket_pair_fit"
        alignment_rmsd = cmd.pair_fit(*fit_pairs)

    else:

        # 若因编号变化导致无法构建局部口袋配对，则使用整体 Cα 拟合作为回退方案
        alignment_method = "whole_protein_super"

        fit_result = cmd.super(
            "rec and polymer.protein and name CA",
            "ref and polymer.protein and name CA"
        )

        alignment_rmsd = fit_result[0]

    # ========================================================
    # 保存对齐后的 receptor 与参考 9IR
    # ========================================================

    aligned_pdb = ALIGNED_DIR / f"{receptor_id}_aligned.pdb"
    reference_ligand_pdb = REFERENCE_LIGAND_DIR / f"{receptor_id}_9IR_reference.pdb"
    config_file = CONFIG_DIR / f"{receptor_id}.box.txt"

    cmd.save(str(aligned_pdb), "rec and polymer.protein")
    cmd.save(str(reference_ligand_pdb), ligand_selection)

    # ========================================================
    # 根据 RFD3 中参考 9IR 坐标计算 docking box
    # ========================================================

    coordinates = cmd.get_coords(ligand_selection)

    center_x = sum(coord[0] for coord in coordinates) / len(coordinates)
    center_y = sum(coord[1] for coord in coordinates) / len(coordinates)
    center_z = sum(coord[2] for coord in coordinates) / len(coordinates)

    span_x = max(coord[0] for coord in coordinates) - min(coord[0] for coord in coordinates)
    span_y = max(coord[1] for coord in coordinates) - min(coord[1] for coord in coordinates)
    span_z = max(coord[2] for coord in coordinates) - min(coord[2] for coord in coordinates)

    # 每个方向给配体增加至少 10 Å 的搜索余量，
    # 同时保证不同 receptor 的 box 至少为 20 Å。
    box_size = max(
        MIN_BOX_SIZE,
        span_x + 10.0,
        span_y + 10.0,
        span_z + 10.0
    )

    config_file.write_text(
        f"center_x = {center_x:.3f}\n"
        f"center_y = {center_y:.3f}\n"
        f"center_z = {center_z:.3f}\n\n"
        f"size_x = {box_size:.3f}\n"
        f"size_y = {box_size:.3f}\n"
        f"size_z = {box_size:.3f}\n",
        encoding="utf-8"
    )

    # ========================================================
    # 记录质量控制信息
    # ========================================================

    results.append({
        "receptor_id": receptor_id,
        "backbone_id": backbone_id,
        "reference_file": str(reference_file),
        "aligned_receptor_pdb": str(aligned_pdb),
        "reference_ligand_pdb": str(reference_ligand_pdb),
        "config_file": str(config_file),
        "alignment_method": alignment_method,
        "pocket_ca_count": pocket_ca_count,
        "alignment_rmsd_A": round(float(alignment_rmsd), 4),
        "ligand_atom_count": ligand_atom_count,
        "center_x": round(center_x, 3),
        "center_y": round(center_y, 3),
        "center_z": round(center_z, 3),
        "box_size": round(box_size, 3),
        "status": "success"
    })

    print(
        f"[SUCCESS] {receptor_id} | "
        f"pocket_CA={pocket_ca_count} | "
        f"RMSD={alignment_rmsd:.4f} Å | "
        f"box={box_size:.2f} Å"
    )

# ============================================================
# 输出 manifest 文件
# ============================================================

manifest_file = QC_DIR / "box_manifest_9IR.csv"

fieldnames = [
    "receptor_id",
    "backbone_id",
    "reference_file",
    "aligned_receptor_pdb",
    "reference_ligand_pdb",
    "config_file",
    "alignment_method",
    "pocket_ca_count",
    "alignment_rmsd_A",
    "ligand_atom_count",
    "center_x",
    "center_y",
    "center_z",
    "box_size",
    "status"
]

with open(manifest_file, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    for row in results:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

success_count = sum(row.get("status") == "success" for row in results)
failed_count = len(results) - success_count

print("")
print("========================================")
print("9IR alignment and box generation finished")
print("========================================")
print(f"Processed receptor count: {len(receptor_files)}")
print(f"Success count: {success_count}")
print(f"Failed count: {failed_count}")
print(f"Manifest file: {manifest_file}")