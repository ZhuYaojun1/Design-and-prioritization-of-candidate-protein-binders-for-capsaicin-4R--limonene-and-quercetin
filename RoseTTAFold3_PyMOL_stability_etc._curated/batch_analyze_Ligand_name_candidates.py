#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch integration and reproducible selection of Ligand_name binder back-predicted structures.

The ligand identity is provided with --ligand-name, allowing the same script to be
used for any ligand project without hard-coding a ligand-specific name.

Expected input files for each backbone:
    buried_X_model_Y.txt or partial_X_model_Y.txt
    buried_X_model_Y_ProtParam_results.xlsx or partial_X_model_Y_ProtParam_results.xlsx
    buried_X_model_Y_RF3_metrics.csv or partial_X_model_Y_RF3_metrics.csv

Outputs are consolidated CSV/TXT/LOG files only; no per-backbone Excel reports
are generated.

Selection hierarchy:
1) Strict_selected:
   stability_prediction == Stable
   ligand_clash_total == 0
   projected_contact_recovery_percent == 100
   Rank: interface recovery desc -> pocket backbone RMSD asc ->
         instability index asc -> usability desc -> solubility desc ->
         overall sequence recovery desc

2) Conditional_selected:
   no strict candidate exists, but at least one candidate is Stable and clash-free
   Rank: contact recovery desc -> interface recovery desc ->
         pocket backbone RMSD asc -> instability index asc ->
         usability desc -> solubility desc -> overall sequence recovery desc

3) Rejected_repair_required:
   no Stable and clash-free candidate exists.
   No representative is admitted into docking/MD. A repair starting point is
   recorded for redesign purposes only.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


NETSOLP_REQUIRED = {"Sequence ID", "solubility", "usability"}
PROTPARAM_REQUIRED = {
    "candidate_id",
    "sequence_recovery",
    "ligand_interface_sequence_recovery",
    "instability_index",
    "stability_prediction",
    "half_life_E_coli_in_vivo",
}
RF3_REQUIRED = {
    "sample_id",
    "pocket_backbone_rmsd_after_CA_fit_A",
    "projected_min_protein_ligand_distance_A",
    "projected_ligand_clash_pair_count_backbone",
    "projected_ligand_clash_pair_count_sidechain",
    "projected_contact_recovery_percent",
}
POCKET_TYPE_CHOICES = ("buried", "partial")
BASE_FROM_TXT_RE = re.compile(
    r"^((?P<pocket_type>buried|partial)_\d+_model_\d+)\.txt$", re.IGNORECASE
)
D_FROM_SEQUENCE_RE = re.compile(r"_b0_d(\d+)$", re.IGNORECASE)
D_FROM_CANDIDATE_RE = re.compile(r"_(\d+)$")


def sanitize_label(text: str) -> str:
    """Create a file-system-friendly ligand label for output names."""
    label = re.sub(r"[^\w.-]+", "_", str(text).strip(), flags=re.UNICODE).strip("._")
    return label or "Ligand_name"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch merge and select one Ligand_name structure per backbone, "
            "supporting buried and partial pocket designs in the same run."
        )
    )
    parser.add_argument(
        "--ligand-name",
        type=str,
        default="Ligand_name",
        help=(
            "Ligand/project name shown in logs and summaries. "
            "Default: Ligand_name."
        ),
    )
    parser.add_argument(
        "--filtering-root",
        type=Path,
        default=Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\5_Filtering_structure"),
        help="Root folder containing ligand-specific input folders.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Folder containing buried/partial triplet files. When omitted, it is set to "
            "<filtering-root>/<ligand-name>/in_<ligand-name>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Default: sibling folder out_<ligand-name>_batch_selection.",
    )
    parser.add_argument(
        "--pocket-types",
        nargs="+",
        choices=POCKET_TYPE_CHOICES,
        default=list(POCKET_TYPE_CHOICES),
        help="Pocket types to process. Default: buried partial.",
    )
    parser.add_argument(
        "--expected-per-backbone",
        type=int,
        default=10,
        help="Expected number of structures per backbone; default is 10.",
    )
    args = parser.parse_args()
    ligand_label = sanitize_label(args.ligand_name)
    if args.input_dir is None:
        args.input_dir = args.filtering_root / ligand_label / f"in_{ligand_label}"
    args.ligand_label = ligand_label
    return args


def setup_logger(output_dir: Path, ligand_label: str) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{ligand_label}_batch_selection_run.log"

    logger = logging.getLogger(f"{ligand_label}_batch_selection")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().strip('"').strip() for c in df.columns]
    return df


def require_columns(df: pd.DataFrame, required: set, source: Path) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source.name} missing columns: {missing}")


def extract_d_index(value: object, regex: re.Pattern, label: str) -> int:
    text = str(value).strip().strip('"')
    match = regex.search(text)
    if not match:
        raise ValueError(f"Unable to extract d-index from {label}: {text}")
    return int(match.group(1))


def read_netsolp(path: Path) -> pd.DataFrame:
    """Read NetSolP txt robustly from tabs or whitespace-formatted text."""
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
        df = clean_headers(df)
        if not NETSOLP_REQUIRED.issubset(df.columns):
            raise ValueError("Tab parsing did not recover expected NetSolP headers.")
    except Exception:
        # Fallback for whitespace-aligned text. Sequence ID contains no whitespace.
        records = []
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        for raw in lines:
            line = raw.strip().replace('"', "")
            if not line or line.lower().startswith("sequence id"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                records.append([parts[0], parts[-2], parts[-1]])
        df = pd.DataFrame(records, columns=["Sequence ID", "solubility", "usability"])

    require_columns(df, NETSOLP_REQUIRED, path)
    df["Sequence ID"] = df["Sequence ID"].astype(str).str.strip().str.strip('"')
    df["d_index"] = df["Sequence ID"].apply(
        lambda x: extract_d_index(x, D_FROM_SEQUENCE_RE, "Sequence ID")
    )
    for col in ["solubility", "usability"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["d_index", "Sequence ID", "solubility", "usability"]]


def read_protparam(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df = clean_headers(df)
    require_columns(df, PROTPARAM_REQUIRED, path)
    df["candidate_id"] = df["candidate_id"].astype(str).str.strip().str.strip('"')
    df["d_index"] = df["candidate_id"].apply(
        lambda x: extract_d_index(x, D_FROM_CANDIDATE_RE, "candidate_id")
    )
    for col in [
        "sequence_recovery",
        "ligand_interface_sequence_recovery",
        "instability_index",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["d_index"] + sorted(PROTPARAM_REQUIRED)]


def read_rf3(path: Path, backbone_id: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = clean_headers(df)
    require_columns(df, RF3_REQUIRED, path)
    df["sample_id"] = pd.to_numeric(df["sample_id"], errors="raise").astype(int)
    df["d_index"] = df["sample_id"] - 1
    df["backbone_id"] = backbone_id

    # Convert numeric-looking RF3 columns without relying on deprecated
    # errors="ignore" behavior in newer pandas versions.
    for col in df.columns:
        if col == "backbone_id":
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() == df[col].notna().sum():
            df[col] = converted
    return df


def discover_batches(
    input_dir: Path, pocket_types: List[str]
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    """Locate complete buried and/or partial input-file triplets."""
    batches: List[Dict[str, object]] = []
    audit: List[Dict[str, str]] = []
    enabled_types = {p.casefold() for p in pocket_types}

    txt_files: List[Path] = []
    for pocket_type in pocket_types:
        txt_files.extend(input_dir.glob(f"{pocket_type}_*_model_*.txt"))
    txt_files = sorted(set(txt_files), key=lambda p: p.name.casefold())

    for txt in txt_files:
        match = BASE_FROM_TXT_RE.match(txt.name)
        if not match:
            continue
        pocket_type = match.group("pocket_type").casefold()
        if pocket_type not in enabled_types:
            continue
        base = match.group(1)
        prot = input_dir / f"{base}_ProtParam_results.xlsx"
        rf3_matches = sorted(
            input_dir.glob(f"{base}_RF3_metrics*.csv"),
            key=lambda p: ("(" in p.name, p.name),
        )
        rf3 = rf3_matches[0] if rf3_matches else None

        complete = prot.exists() and rf3 is not None
        audit.append(
            {
                "pocket_type": pocket_type,
                "backbone_id": base,
                "netsolp_file": txt.name,
                "protparam_file": prot.name if prot.exists() else "",
                "rf3_file": rf3.name if rf3 else "",
                "triplet_complete": "Yes" if complete else "No",
                "audit_note": (
                    "" if complete else "Missing ProtParam workbook and/or RF3 metrics CSV."
                ),
            }
        )
        if complete:
            batches.append(
                {
                    "pocket_type": pocket_type,
                    "backbone_id": base,
                    "txt": txt,
                    "prot": prot,
                    "rf3": rf3,
                }
            )

    return batches, audit


def merge_batch(
    batch: Dict[str, object], expected_rows: int, logger: logging.Logger
) -> pd.DataFrame:
    base = str(batch["backbone_id"])
    pocket_type = str(batch["pocket_type"])
    netsolp = read_netsolp(Path(batch["txt"]))
    prot = read_protparam(Path(batch["prot"]))
    rf3 = read_rf3(Path(batch["rf3"]), base)

    for name, table in [("NetSolP", netsolp), ("ProtParam", prot), ("RF3", rf3)]:
        if table["d_index"].duplicated().any():
            raise ValueError(f"{base}: duplicated d_index in {name} table.")

    merged = rf3.merge(prot, on="d_index", how="outer", validate="one_to_one")
    merged = merged.merge(netsolp, on="d_index", how="outer", validate="one_to_one")
    merged["backbone_id"] = base
    merged["pocket_type"] = pocket_type

    if len(merged) != expected_rows:
        logger.warning(
            "[%s] Expected %d candidates but merged %d rows.",
            base, expected_rows, len(merged)
        )

    essential = [
        "candidate_id",
        "Sequence ID",
        "sample_id",
        "sequence_recovery",
        "ligand_interface_sequence_recovery",
        "instability_index",
        "stability_prediction",
        "pocket_backbone_rmsd_after_CA_fit_A",
        "projected_ligand_clash_pair_count_backbone",
        "projected_ligand_clash_pair_count_sidechain",
        "projected_contact_recovery_percent",
        "solubility",
        "usability",
    ]
    incomplete = merged[essential].isna().any(axis=1)
    if incomplete.any():
        bad = merged.loc[incomplete, "d_index"].tolist()
        raise ValueError(f"{base}: missing merged metrics for d-index values {bad}.")

    merged["ligand_clash_total"] = (
        pd.to_numeric(merged["projected_ligand_clash_pair_count_backbone"])
        + pd.to_numeric(merged["projected_ligand_clash_pair_count_sidechain"])
    )
    merged["is_stable"] = (
        merged["stability_prediction"].astype(str).str.strip().str.casefold() == "stable"
    )
    merged["is_clash_free"] = merged["ligand_clash_total"] == 0
    merged["is_full_contact_recovery"] = np.isclose(
        merged["projected_contact_recovery_percent"].astype(float), 100.0, atol=1e-6
    )
    merged["strict_gate_pass"] = (
        merged["is_stable"] & merged["is_clash_free"] & merged["is_full_contact_recovery"]
    )
    merged["conditional_gate_pass"] = merged["is_stable"] & merged["is_clash_free"]
    return merged


def sort_selection_pool(pool: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "strict":
        columns = [
            "ligand_interface_sequence_recovery",
            "pocket_backbone_rmsd_after_CA_fit_A",
            "instability_index",
            "usability",
            "solubility",
            "sequence_recovery",
        ]
        ascending = [False, True, True, False, False, False]
    elif mode == "conditional":
        columns = [
            "projected_contact_recovery_percent",
            "ligand_interface_sequence_recovery",
            "pocket_backbone_rmsd_after_CA_fit_A",
            "instability_index",
            "usability",
            "solubility",
            "sequence_recovery",
        ]
        ascending = [False, False, True, True, False, False, False]
    else:
        raise ValueError(f"Unsupported selection mode: {mode}")
    ranked = pool.sort_values(columns, ascending=ascending, kind="mergesort").copy()
    ranked["selection_pool_rank"] = range(1, len(ranked) + 1)
    return ranked


def rank_repair_pool(group: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    clash_free = group[group["is_clash_free"]].copy()
    stable = group[group["is_stable"]].copy()

    if not clash_free.empty:
        # Rejection in this branch means clash-free candidates are not Stable:
        # protect existing clash-free pocket while repairing stability.
        focus = "Stability_optimization_of_clash_free_candidate"
        ranked = clash_free.sort_values(
            [
                "projected_contact_recovery_percent",
                "ligand_interface_sequence_recovery",
                "pocket_backbone_rmsd_after_CA_fit_A",
                "instability_index",
                "usability",
            ],
            ascending=[False, False, True, True, False],
            kind="mergesort",
        ).copy()
    elif not stable.empty:
        focus = "Clash_removal_from_stable_candidate"
        ranked = stable.sort_values(
            [
                "ligand_clash_total",
                "projected_contact_recovery_percent",
                "ligand_interface_sequence_recovery",
                "pocket_backbone_rmsd_after_CA_fit_A",
                "instability_index",
                "usability",
            ],
            ascending=[True, False, False, True, True, False],
            kind="mergesort",
        ).copy()
    else:
        focus = "Redesign_required_no_stable_candidate"
        ranked = group.sort_values(
            [
                "ligand_clash_total",
                "projected_contact_recovery_percent",
                "ligand_interface_sequence_recovery",
                "pocket_backbone_rmsd_after_CA_fit_A",
                "instability_index",
                "usability",
            ],
            ascending=[True, False, False, True, True, False],
            kind="mergesort",
        ).copy()

    ranked["selection_pool_rank"] = range(1, len(ranked) + 1)
    return ranked, focus


def risk_flags(row: pd.Series, status: str) -> str:
    flags: List[str] = []
    contact = float(row["projected_contact_recovery_percent"])
    rmsd = float(row["pocket_backbone_rmsd_after_CA_fit_A"])
    if status == "Conditional_selected":
        if contact < 90:
            flags.append("contact_recovery_below_90")
        if rmsd > 1.0:
            flags.append("pocket_RMSD_above_1A")
    return ";".join(flags) if flags else "None"


def assess_backbone(group: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, Dict[str, object]]:
    backbone = str(group["backbone_id"].iloc[0])
    pocket_type = str(group["pocket_type"].iloc[0])
    strict = group[group["strict_gate_pass"]]
    conditional = group[group["conditional_gate_pass"]]

    group = group.copy()
    group["accepted_representative"] = False
    group["decision_status"] = ""
    group["selection_pool_rank"] = np.nan
    group["decision_note"] = ""

    selected: Optional[pd.Series] = None
    repair: Optional[pd.Series] = None
    alternative: Optional[pd.Series] = None
    repair_focus = ""

    if not strict.empty:
        ranked = sort_selection_pool(strict, "strict")
        selected = ranked.iloc[0]
        status = "Strict_selected"
        reason = (
            "Stable; clash-free; 100% contact recovery; ranked within strict pool by "
            "interface recovery, pocket RMSD, instability, usability, solubility and sequence recovery."
        )
        pool_kind = "Strict"
    elif not conditional.empty:
        ranked = sort_selection_pool(conditional, "conditional")
        selected = ranked.iloc[0]
        status = "Conditional_selected"
        reason = (
            "No strict candidate; selected from Stable clash-free pool by contact recovery, "
            "interface recovery, pocket RMSD, instability, usability, solubility and sequence recovery."
        )
        pool_kind = "Conditional"
    else:
        ranked, repair_focus = rank_repair_pool(group)
        repair = ranked.iloc[0]
        alternative = ranked.iloc[1] if len(ranked) > 1 else None
        status = "Rejected_repair_required"
        reason = (
            "No Stable and clash-free candidate exists; backbone excluded from accepted docking set. "
            "Repair candidates are recorded for redesign only."
        )
        pool_kind = "Repair"

    group["decision_status"] = status
    group["decision_note"] = reason

    for idx, rank_row in ranked.iterrows():
        group.loc[idx, "selection_pool_rank"] = rank_row["selection_pool_rank"]

    if selected is not None:
        group.loc[selected.name, "accepted_representative"] = True
        risk = risk_flags(selected, status)
        md_priority = (
            "Docking_and_MD_candidate"
            if status == "Strict_selected"
            else "Docking_first_review_before_MD"
        )
        record = {
            "pocket_type": pocket_type,
            "backbone_id": backbone,
            "decision_status": status,
            "accepted_candidate_id": selected["candidate_id"],
            "accepted_sequence_id": selected["Sequence ID"],
            "sample_id": int(selected["sample_id"]),
            "strict_candidate_count": int(len(strict)),
            "conditional_candidate_count": int(len(conditional)),
            "selection_pool": pool_kind,
            "selection_reason": reason,
            "risk_flags": risk,
            "md_priority": md_priority,
            "repair_focus": "",
            "repair_candidate_id": "",
            "repair_alternative_id": "",
            "ligand_clash_total": int(selected["ligand_clash_total"]),
            "contact_recovery_percent": float(selected["projected_contact_recovery_percent"]),
            "ligand_interface_sequence_recovery": float(selected["ligand_interface_sequence_recovery"]),
            "sequence_recovery": float(selected["sequence_recovery"]),
            "pocket_backbone_rmsd_A": float(selected["pocket_backbone_rmsd_after_CA_fit_A"]),
            "instability_index": float(selected["instability_index"]),
            "stability_prediction": str(selected["stability_prediction"]),
            "half_life_E_coli_in_vivo": str(selected["half_life_E_coli_in_vivo"]),
            "solubility": float(selected["solubility"]),
            "usability": float(selected["usability"]),
        }
        logger.info(
            "[%s] %s -> %s | contact=%.4f | interface=%.4f | RMSD=%.4f | risks=%s",
            backbone, status, selected["candidate_id"],
            selected["projected_contact_recovery_percent"],
            selected["ligand_interface_sequence_recovery"],
            selected["pocket_backbone_rmsd_after_CA_fit_A"],
            risk,
        )
    else:
        assert repair is not None
        record = {
            "pocket_type": pocket_type,
            "backbone_id": backbone,
            "decision_status": status,
            "accepted_candidate_id": "",
            "accepted_sequence_id": "",
            "sample_id": "",
            "strict_candidate_count": int(len(strict)),
            "conditional_candidate_count": int(len(conditional)),
            "selection_pool": pool_kind,
            "selection_reason": reason,
            "risk_flags": "not_accepted",
            "md_priority": "Do_not_dock_until_repaired",
            "repair_focus": repair_focus,
            "repair_candidate_id": repair["candidate_id"],
            "repair_alternative_id": (
                alternative["candidate_id"] if alternative is not None else ""
            ),
            "ligand_clash_total": int(repair["ligand_clash_total"]),
            "contact_recovery_percent": float(repair["projected_contact_recovery_percent"]),
            "ligand_interface_sequence_recovery": float(repair["ligand_interface_sequence_recovery"]),
            "sequence_recovery": float(repair["sequence_recovery"]),
            "pocket_backbone_rmsd_A": float(repair["pocket_backbone_rmsd_after_CA_fit_A"]),
            "instability_index": float(repair["instability_index"]),
            "stability_prediction": str(repair["stability_prediction"]),
            "half_life_E_coli_in_vivo": str(repair["half_life_E_coli_in_vivo"]),
            "solubility": float(repair["solubility"]),
            "usability": float(repair["usability"]),
        }
        logger.warning(
            "[%s] REJECTED; repair=%s; alternative=%s; focus=%s | clashes=%d | contact=%.4f | RMSD=%.4f",
            backbone,
            repair["candidate_id"],
            alternative["candidate_id"] if alternative is not None else "",
            repair_focus,
            repair["ligand_clash_total"],
            repair["projected_contact_recovery_percent"],
            repair["pocket_backbone_rmsd_after_CA_fit_A"],
        )

    return group, record


def describe_selected(df: pd.DataFrame, column: str) -> str:
    if df.empty:
        return "NA"
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return "NA"
    return f"{series.median():.4f} (median; range {series.min():.4f}-{series.max():.4f})"


def write_results_summaries(
    output_dir: Path,
    decisions: pd.DataFrame,
    all_candidates: pd.DataFrame,
    ligand_name: str,
) -> None:
    strict = decisions[decisions["decision_status"] == "Strict_selected"]
    conditional = decisions[decisions["decision_status"] == "Conditional_selected"]
    rejected = decisions[decisions["decision_status"] == "Rejected_repair_required"]
    accepted = decisions[decisions["decision_status"].isin(["Strict_selected", "Conditional_selected"])]

    n_backbones = len(decisions)
    n_candidates = len(all_candidates)
    n_accepted = len(accepted)

    pocket_counts = decisions["pocket_type"].value_counts().to_dict()
    pocket_cn = "；".join(f"{key} 型 {value} 个" for key, value in sorted(pocket_counts.items())) or "None"
    pocket_en = "; ".join(f"{key}: {value}" for key, value in sorted(pocket_counts.items())) or "None"

    strict_ids = ", ".join(strict["accepted_candidate_id"].astype(str).tolist()) or "None"
    conditional_ids = ", ".join(conditional["accepted_candidate_id"].astype(str).tolist()) or "None"
    rejected_ids = ", ".join(rejected["backbone_id"].astype(str).tolist()) or "None"

    cn = f"""{ligand_name} 回溯结构批量筛选结果摘要
生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

共纳入 {n_backbones} 个 {ligand_name} 候选骨架（{pocket_cn}），整合评估 {n_candidates} 个 LigandMPNN 序列回溯结构。本次分析可在同一次运行中同时处理 buried 与 partial 两类结合口袋设计。筛选时首先要求候选结构具有稳定性预测为 Stable 且不存在预测配体冲突；在此基础上，接触恢复率达到 100% 的结构被定义为严格入选。结果显示，{len(strict)} 个骨架获得严格入选代表结构，{len(conditional)} 个骨架仅获得无冲突但接触恢复不完整的条件性代表结构，另有 {len(rejected)} 个骨架因不存在 Stable 且无配体冲突的候选结构而被归入需修复/重新设计类别。最终，共有 {n_accepted} 个代表结构可进入后续 docking 初筛。

严格入选代表结构的 pocket backbone RMSD 为 {describe_selected(strict, "pocket_backbone_rmsd_A")} Å，界面序列恢复率为 {describe_selected(strict, "ligand_interface_sequence_recovery")}；条件性入选代表结构的 pocket backbone RMSD 为 {describe_selected(conditional, "pocket_backbone_rmsd_A")} Å，接触恢复率为 {describe_selected(conditional, "contact_recovery_percent")}%。条件性入选结构应在 docking 后重新审查配体姿势和关键接触，再决定是否进入较长时间尺度的分子动力学模拟。

严格入选结构：
{strict_ids}

条件性入选结构：
{conditional_ids}

拒绝/需修复骨架：
{rejected_ids}
"""

    en = f"""Batch screening summary for back-predicted {ligand_name}-binding structures
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

A total of {n_backbones} candidate {ligand_name}-binding backbones ({pocket_en}), comprising {n_candidates} LigandMPNN-designed and structure-back-predicted candidates, were integrated and evaluated. Both buried and partial pocket-design classes can be processed in the same run. Candidates predicted to be Stable and free of projected ligand clashes were first retained. Among them, structures achieving 100% recovery of the original designed ligand-contact positions were defined as strict representatives. Accordingly, {len(strict)} backbones yielded a strict representative, {len(conditional)} backbones yielded a conditional clash-free representative with incomplete contact recovery, and {len(rejected)} backbones were classified as requiring repair or redesign because no Stable clash-free candidate was available. In total, {n_accepted} representative structures were retained for downstream docking screening.

For strict representatives, the pocket backbone RMSD was {describe_selected(strict, "pocket_backbone_rmsd_A")} Å and the ligand-interface sequence recovery was {describe_selected(strict, "ligand_interface_sequence_recovery")}. For conditional representatives, the pocket backbone RMSD was {describe_selected(conditional, "pocket_backbone_rmsd_A")} Å and the contact recovery rate was {describe_selected(conditional, "contact_recovery_percent")}%. Conditional representatives should be re-evaluated after docking before being advanced to longer molecular dynamics simulations.

Strict representatives:
{strict_ids}

Conditional representatives:
{conditional_ids}

Rejected or repair-required backbones:
{rejected_ids}
"""

    (output_dir / "06_results_summary_CN.txt").write_text(cn, encoding="utf-8-sig")
    (output_dir / "07_results_summary_EN.txt").write_text(en, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    ligand_name = args.ligand_name
    ligand_label = args.ligand_label
    input_dir = args.input_dir
    output_dir = args.output_dir or (input_dir.parent / f"out_{ligand_label}_batch_selection")
    logger = setup_logger(output_dir, ligand_label)

    logger.info("Ligand name: %s", ligand_name)
    logger.info("Pocket types included: %s", ", ".join(args.pocket_types))
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)
    if not input_dir.exists():
        logger.error("Input directory does not exist: %s", input_dir)
        return 1

    batches, audit = discover_batches(input_dir, args.pocket_types)
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(output_dir / "00_batch_file_audit.csv", index=False, encoding="utf-8-sig")

    logger.info(
        "Discovered %d complete triplets and %d TXT-defined batches.",
        len(batches), len(audit)
    )
    if not batches:
        logger.error("No complete file triplets were found for the selected pocket type(s).")
        return 1

    all_groups: List[pd.DataFrame] = []
    decision_records: List[Dict[str, object]] = []
    failed: List[Dict[str, str]] = []

    for batch in batches:
        base = str(batch["backbone_id"])
        try:
            merged = merge_batch(batch, args.expected_per_backbone, logger)
            annotated, decision = assess_backbone(merged, logger)
            all_groups.append(annotated)
            decision_records.append(decision)
        except Exception as exc:
            logger.exception("[%s] Processing failed: %s", base, exc)
            failed.append({"backbone_id": base, "error": str(exc)})

    if not decision_records:
        logger.error("No batches were successfully processed.")
        pd.DataFrame(failed).to_csv(
            output_dir / "00_processing_failures.csv", index=False, encoding="utf-8-sig"
        )
        return 1

    all_candidates = pd.concat(all_groups, ignore_index=True)
    decisions = (
        pd.DataFrame(decision_records)
        .sort_values(["pocket_type", "backbone_id"])
        .reset_index(drop=True)
    )
    accepted = decisions[
        decisions["decision_status"].isin(["Strict_selected", "Conditional_selected"])
    ].copy()
    rejected = decisions[decisions["decision_status"] == "Rejected_repair_required"].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    all_candidates.to_csv(
        output_dir / "01_all_candidates_merged_and_annotated.csv",
        index=False, encoding="utf-8-sig"
    )
    decisions.to_csv(
        output_dir / "02_backbone_decisions.csv",
        index=False, encoding="utf-8-sig"
    )
    accepted.to_csv(
        output_dir / "03_accepted_representatives_for_docking.csv",
        index=False, encoding="utf-8-sig"
    )
    rejected.to_csv(
        output_dir / "04_rejected_backbones_repair_targets.csv",
        index=False, encoding="utf-8-sig"
    )

    accepted_ids = accepted["accepted_candidate_id"].astype(str).tolist()
    (output_dir / "05_accepted_candidate_ids.txt").write_text(
        "\n".join(accepted_ids) + ("\n" if accepted_ids else ""),
        encoding="utf-8-sig"
    )

    write_results_summaries(output_dir, decisions, all_candidates, ligand_name)

    if failed:
        pd.DataFrame(failed).to_csv(
            output_dir / "00_processing_failures.csv", index=False, encoding="utf-8-sig"
        )
        logger.warning("%d batches failed; see 00_processing_failures.csv.", len(failed))

    counts = decisions["decision_status"].value_counts().to_dict()
    pocket_counts = decisions["pocket_type"].value_counts().to_dict()
    logger.info("Completed successfully: %d backbones, %d candidate structures.", len(decisions), len(all_candidates))
    logger.info("Pocket type counts: %s", pocket_counts)
    logger.info("Status counts: %s", counts)
    logger.info("Accepted representatives for docking: %d", len(accepted))
    logger.info("Output files written to: %s", output_dir)

    print(f"\n=== {ligand_name} batch selection complete ===")
    print(f"Pocket types included: {', '.join(args.pocket_types)}")
    print(f"Backbones processed: {len(decisions)}")
    print(f"Backbones by pocket type: {pocket_counts}")
    print(f"Candidate structures merged: {len(all_candidates)}")
    print(f"Strict selected: {counts.get('Strict_selected', 0)}")
    print(f"Conditional selected: {counts.get('Conditional_selected', 0)}")
    print(f"Rejected / repair required: {counts.get('Rejected_repair_required', 0)}")
    print(f"Accepted for docking screening: {len(accepted)}")
    print(f"Results folder: {output_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
