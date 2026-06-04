from pathlib import Path
import csv
import time
from pymol import cmd


# ============================================================
# 1. 路径设置
# ============================================================

INPUT_CSV = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3\9IR_output\all\JSON\ranking_json\rfd3_json_pass.csv"
)

STRUCTURE_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3\9IR_output\all\CIF"
)

OUTPUT_DIR = Path(
    r"C:\Users\Lenovo\Desktop\small_molecule_design\2_RFD3\9IR_output\ranking"
)

OUT_ALL = OUTPUT_DIR / "9IR_pymol_all_ranking_fixed.csv"
OUT_TOP20 = OUTPUT_DIR / "9IR_pymol_top20_fixed.csv"
OUT_LOG = OUTPUT_DIR / "9IR_pymol_filter_log_fixed.txt"


# ============================================================
# 2. 9IR_buried 参数
# ============================================================

# 9IR 的重原子：C,C1,C2,C3,C4,C5,C6,C7,C8,C9
# buried 模式中，这 10 个原子都应作为 buried atoms。
DEFAULT_9IR_BURIED_ATOMS = [
    "C", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"
]

EXPECTED_9IR_HEAVY_ATOMS = 10


# ============================================================
# 3. 基础函数
# ============================================================

def log_print(text):
    print(text, flush=True)
    with open(OUT_LOG, "a", encoding="utf-8") as f:
        f.write(str(text) + "\n")


def parse_atom_list(text):
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def to_float(x, default=999.0):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=999):
    try:
        return int(float(x))
    except Exception:
        return default


def find_structure(structure_dir, json_file_name, example_id):
    """
    根据 json_file_name 或 example_id 在 structure_dir 中寻找对应 CIF/PDB。
    例如：
    9IR_9IR_buried_2_model_6.json
    对应：
    9IR_9IR_buried_2_model_6.cif
    """
    structure_dir = Path(structure_dir)
    json_stem = Path(json_file_name).stem

    candidates = [
        structure_dir / f"{json_stem}.cif",
        structure_dir / f"{json_stem}.pdb",
        structure_dir / f"{example_id}.cif",
        structure_dir / f"{example_id}.pdb",
    ]

    for c in candidates:
        if c.exists():
            return c

    patterns = [
        f"{json_stem}*.cif",
        f"{json_stem}*.pdb",
        f"{example_id}*.cif",
        f"{example_id}*.pdb",
    ]

    for pattern in patterns:
        hits = list(structure_dir.glob(pattern))
        if hits:
            return hits[0]

    return None


def safe_sort_residues(residue_set):
    """
    pocket residue 排序。
    避免某些 resi 不是纯数字时排序报错。
    """
    def key_func(x):
        chain, resi, resn = x
        digits = "".join([c for c in str(resi) if c.isdigit()])
        num = int(digits) if digits else 999999
        return (chain, num, resn)

    return sorted(residue_set, key=key_func)


# ============================================================
# 4. 9IR 专用评分函数
# ============================================================

def score_candidate(row):
    """
    9IR_buried 综合评分。
    分数越低越好。

    注意：
    9IR 是小型疏水单萜类小分子。
    buried 模式下不设置 exposed atoms，因此不对 exposed_fraction 做惩罚。
    """

    json_score = to_float(row.get("json_score", 9999))

    ligand_count = to_int(row.get("ligand_count", 0), 0)
    buried_fraction = to_float(row.get("buried_fraction", 0), 0)
    pocket4 = to_int(row.get("pocket4_residue_count", 0), 0)
    pocket5 = to_int(row.get("pocket5_residue_count", 0), 0)
    severe_clash = to_int(row.get("severe_clash_1p8", 999), 999)
    close_overlap = to_int(row.get("close_overlap_2p2", 999), 999)

    penalty = 0.0

    # 配体缺失或不完整，强烈惩罚
    if ligand_count == 0:
        penalty += 5000

    if ligand_count < EXPECTED_9IR_HEAVY_ATOMS:
        penalty += 1000

    # 严重空间冲突
    if severe_clash > 0:
        penalty += 2000 + 200 * severe_clash

    # 2.2 Å 内近距离接触过多也要惩罚，但比 1.8 Å 温和
    if close_overlap > 0:
        penalty += 50 * close_overlap

    # 9IR_buried 要求 9IR 尽量被疏水 pocket 包围
    if buried_fraction < 0.80:
        penalty += 800 * (0.80 - buried_fraction)

    # 9IR 是小分子，pocket4 不需要像 4DY 那么大
    if pocket4 < 6:
        penalty += 80 * (6 - pocket4)

    if pocket5 < 8:
        penalty += 30 * (8 - pocket5)

    return round(json_score + penalty, 4)


def classify_pymol(row):
    """
    9IR_buried 专用 PyMOL 分级。
    """

    ligand_count = to_int(row.get("ligand_count", 0), 0)
    buried_fraction = to_float(row.get("buried_fraction", 0), 0)
    pocket4 = to_int(row.get("pocket4_residue_count", 0), 0)
    severe_clash = to_int(row.get("severe_clash_1p8", 999), 999)

    if ligand_count == 0:
        return "Fail_ligand_missing"

    if ligand_count < EXPECTED_9IR_HEAVY_ATOMS:
        return "Fail_ligand_incomplete"

    if severe_clash > 0:
        return "Fail_severe_clash"

    if buried_fraction < 0.60:
        return "Fail_low_buried_contact"

    if pocket4 < 4:
        return "Fail_no_clear_pocket"

    if buried_fraction >= 0.90 and pocket4 >= 6:
        return "Excellent_hydrophobic_pocket"

    if buried_fraction >= 0.80 and pocket4 >= 6:
        return "Good_hydrophobic_pocket"

    return "Acceptable_manual_check"


# ============================================================
# 5. 主筛选函数
# ============================================================

def run_filter():
    start_time = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write("PyMOL ligand-based filtering log for 9IR_buried\n")

    log_print("\n========== PyMOL 9IR-buried ligand-based filtering started ==========")
    log_print(f"Input CSV: {INPUT_CSV}")
    log_print(f"Structure dir: {STRUCTURE_DIR}")
    log_print(f"Output all CSV: {OUT_ALL}")
    log_print(f"Output top20 CSV: {OUT_TOP20}")
    log_print("====================================================================\n")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    if not STRUCTURE_DIR.exists():
        raise FileNotFoundError(f"Structure directory not found: {STRUCTURE_DIR}")

    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No rows found in input CSV: {INPUT_CSV}")

    total = len(rows)
    log_print(f"Total candidates to process: {total}\n")

    results = []

    for idx, row in enumerate(rows, start=1):
        json_file_name = row.get("json_file_name", "")
        example_id = row.get("example_id", "")
        ligand = row.get("ligand", "")

        structure_file = find_structure(
            structure_dir=STRUCTURE_DIR,
            json_file_name=json_file_name,
            example_id=example_id,
        )

        log_print(
            f"[{idx}/{total}] Processing {json_file_name} | "
            f"example_id={example_id} | ligand={ligand}"
        )

        if structure_file is None:
            row["pymol_status"] = "structure_not_found"
            row["structure_file"] = ""
            row["protein_count"] = 0
            row["ligand_count"] = 0
            row["buried_total"] = 0
            row["buried_contact"] = 0
            row["buried_fraction"] = 0
            row["pocket4_residue_count"] = 0
            row["pocket5_residue_count"] = 0
            row["pocket4_residue_list"] = ""
            row["pocket5_residue_list"] = ""
            row["severe_clash_1p8"] = 999
            row["close_overlap_2p2"] = 999
            row["pymol_grade"] = "Fail_structure_not_found"
            row["final_score"] = 999999

            log_print("    Status: structure not found\n")
            results.append(row)
            continue

        obj = f"m{idx:05d}"

        # 使用不容易与 PyMOL 关键词冲突的选择名
        prot_name = f"sel_prot_{idx}"
        lig_name = f"sel_lig_{idx}"
        buried_name = f"sel_buried_{idx}"
        pocket4_name = f"sel_pocket4_{idx}"
        pocket5_name = f"sel_pocket5_{idx}"
        severe_name = f"sel_severe_{idx}"
        close_name = f"sel_close_{idx}"

        try:
            cmd.delete("all")
            cmd.load(str(structure_file), obj)

            protein_sel = f"({obj} and chain A)"
            lig_sel = f"({obj} and chain B and resn {ligand})"

            # 如果 chain B 选不到 ligand，退一步只按 resn 选择
            if cmd.count_atoms(lig_sel) == 0:
                lig_sel = f"({obj} and resn {ligand})"

            cmd.select(prot_name, protein_sel)
            cmd.select(lig_name, lig_sel)

            ligand_count = cmd.count_atoms(f"({lig_name}) and not hydro")
            protein_count = cmd.count_atoms(f"({prot_name}) and not hydro")

            # 优先读取 CSV 中 buried_atoms；如果为空，则使用 9IR 默认 10 个重原子
            buried_atoms = parse_atom_list(row.get("buried_atoms", ""))
            if not buried_atoms:
                buried_atoms = DEFAULT_9IR_BURIED_ATOMS

            buried_sel = f"({lig_name}) and name " + "+".join(buried_atoms)
            cmd.select(buried_name, buried_sel)

            buried_total = cmd.count_atoms(buried_name)

            cmd.select(
                "buried_contact",
                f"({buried_name}) within 4.0 of ({prot_name})"
            )
            buried_contact = cmd.count_atoms("buried_contact")

            buried_fraction = buried_contact / buried_total if buried_total else 0

            # pocket residues
            cmd.select(
                pocket4_name,
                f"byres (({prot_name}) within 4.0 of ({lig_name}))"
            )

            cmd.select(
                pocket5_name,
                f"byres (({prot_name}) within 5.0 of ({lig_name}))"
            )

            pocket4_residues = []
            pocket5_residues = []

            cmd.iterate(
                f"{pocket4_name} and name CA",
                "pocket4_residues.append((chain, resi, resn))",
                space={"pocket4_residues": pocket4_residues},
            )

            cmd.iterate(
                f"{pocket5_name} and name CA",
                "pocket5_residues.append((chain, resi, resn))",
                space={"pocket5_residues": pocket5_residues},
            )

            pocket4_set = safe_sort_residues(set(pocket4_residues))
            pocket5_set = safe_sort_residues(set(pocket5_residues))

            pocket4_list = ";".join([f"{r[2]}{r[1]}" for r in pocket4_set])
            pocket5_list = ";".join([f"{r[2]}{r[1]}" for r in pocket5_set])

            # clashes
            cmd.select(
                severe_name,
                f"(({prot_name}) and not hydro) within 1.8 of (({lig_name}) and not hydro)"
            )

            cmd.select(
                close_name,
                f"(({prot_name}) and not hydro) within 2.2 of (({lig_name}) and not hydro)"
            )

            severe_clash_1p8 = cmd.count_atoms(severe_name)
            close_overlap_2p2 = cmd.count_atoms(close_name)

            row["pymol_status"] = "ok"
            row["structure_file"] = str(structure_file)

            row["protein_count"] = protein_count
            row["ligand_count"] = ligand_count

            row["buried_total"] = buried_total
            row["buried_contact"] = buried_contact
            row["buried_fraction"] = round(buried_fraction, 4)

            # 9IR_buried 没有 exposed atom
            row["exposed_total"] = 0
            row["exposed_contact"] = 0
            row["exposed_fraction"] = 0

            row["pocket4_residue_count"] = len(pocket4_set)
            row["pocket5_residue_count"] = len(pocket5_set)
            row["pocket4_residue_list"] = pocket4_list
            row["pocket5_residue_list"] = pocket5_list

            row["severe_clash_1p8"] = severe_clash_1p8
            row["close_overlap_2p2"] = close_overlap_2p2

            row["pymol_grade"] = classify_pymol(row)
            row["final_score"] = score_candidate(row)

            log_print(
                f"    protein_atoms={protein_count} | "
                f"lig={ligand_count} | "
                f"buried={buried_contact}/{buried_total} "
                f"({buried_fraction:.2f}) | "
                f"pocket4={len(pocket4_set)} | "
                f"pocket5={len(pocket5_set)} | "
                f"clash1.8={severe_clash_1p8} | "
                f"grade={row['pymol_grade']} | "
                f"final_score={row['final_score']}"
            )
            log_print("")

            results.append(row)

        except Exception as e:
            row["pymol_status"] = f"error: {e}"
            row["structure_file"] = str(structure_file)
            row["protein_count"] = 0
            row["ligand_count"] = 0
            row["buried_total"] = 0
            row["buried_contact"] = 0
            row["buried_fraction"] = 0
            row["exposed_total"] = 0
            row["exposed_contact"] = 0
            row["exposed_fraction"] = 0
            row["pocket4_residue_count"] = 0
            row["pocket5_residue_count"] = 0
            row["pocket4_residue_list"] = ""
            row["pocket5_residue_list"] = ""
            row["severe_clash_1p8"] = 999
            row["close_overlap_2p2"] = 999
            row["pymol_grade"] = "Fail_script_error"
            row["final_score"] = 999999

            log_print(f"    ERROR: {e}\n")
            results.append(row)

    # 排序
    results = sorted(results, key=lambda r: to_float(r.get("final_score", 999999)))

    for i, row in enumerate(results, start=1):
        row["final_rank"] = i

    fieldnames = list(results[0].keys())

    with open(OUT_ALL, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(OUT_TOP20, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results[:20])

    # 打印 Top 20
    log_print("\n========== PyMOL 9IR-buried final Top 20 ==========\n")

    header = (
        f"{'rank':>4}  {'json_file':<34}  {'json_score':>10}  "
        f"{'final_score':>11}  {'buried':>9}  "
        f"{'pocket4':>7}  {'pocket5':>7}  {'clash1.8':>8}  {'grade':<30}"
    )

    log_print(header)
    log_print("-" * len(header))

    for row in results[:20]:
        buried_text = f"{row.get('buried_contact')}/{row.get('buried_total')}"

        log_print(
            f"{row.get('final_rank'):>4}  "
            f"{row.get('json_file_name',''):<34}  "
            f"{to_float(row.get('json_score', 9999)):>10.4f}  "
            f"{to_float(row.get('final_score', 9999)):>11.4f}  "
            f"{buried_text:>9}  "
            f"{row.get('pocket4_residue_count'):>7}  "
            f"{row.get('pocket5_residue_count'):>7}  "
            f"{row.get('severe_clash_1p8'):>8}  "
            f"{row.get('pymol_grade',''):<30}"
        )

    elapsed = time.time() - start_time

    log_print("\n========== Output Files ==========")
    log_print(f"Full PyMOL ranking saved to: {OUT_ALL}")
    log_print(f"Top 20 PyMOL ranking saved to: {OUT_TOP20}")
    log_print(f"Log file saved to: {OUT_LOG}")
    log_print(f"Elapsed time: {elapsed:.2f} seconds")
    log_print("No FASTA file was generated.")
    log_print("========== Filtering finished ==========\n")


# ============================================================
# 6. 在 PyMOL 中运行
# ============================================================

run_filter()