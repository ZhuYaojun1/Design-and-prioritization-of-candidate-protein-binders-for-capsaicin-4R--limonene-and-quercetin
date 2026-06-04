# batch_cif_to_pdb.py
# 在 PyMOL 命令行中执行：run batch_cif_to_pdb.py

from pymol import cmd
from pathlib import Path

# =========================
# 1. 设置根目录
# =========================
base_input_dir = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\3_LigandMPNN_NEW\Input_CIF")
base_output_dir = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\3_LigandMPNN_NEW\Input_PDB")

# 确保输出根目录存在
base_output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2. PyMOL 输出设置
# =========================
cmd.set("pdb_use_ter_records", 1)   # 保留 TER 记录，区分链
cmd.set("retain_order", 1)          # 保持原子顺序

# =========================
# 3. 获取所有需要处理的子文件夹
# =========================
subdirs = [d for d in base_input_dir.iterdir() if d.is_dir()]

if not subdirs:
    raise FileNotFoundError(f"未在 {base_input_dir} 中找到任何子文件夹")

print(f"找到 {len(subdirs)} 个子文件夹: {[d.name for d in subdirs]}")

# =========================
# 4. 逐个处理子文件夹
# =========================
for subdir in subdirs:
    input_dir = subdir
    output_dir = base_output_dir / subdir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(input_dir.glob("*.cif"))
    print(f"\n处理子文件夹: {subdir.name}  (找到 {len(cif_files)} 个 CIF 文件)")

    if not cif_files:
        print("  跳过（该文件夹中没有 .cif 文件）")
        continue

    for i, cif_file in enumerate(cif_files, start=1):
        # PyMOL 对象名：子文件夹名 + 序号，避免冲突
        obj_name = f"{subdir.name}_model_{i:03d}"
        pdb_file = output_dir / f"{cif_file.stem}.pdb"

        print(f"  [{i}/{len(cif_files)}] {cif_file.name} -> {pdb_file.name}")

        # 清空当前场景，加载 CIF，保存为 PDB
        cmd.delete("all")
        cmd.load(str(cif_file), obj_name)
        cmd.save(str(pdb_file), obj_name, state=1)

print("\n所有子文件夹转换完成！")
print(f"PDB 文件已保存至: {base_output_dir}")