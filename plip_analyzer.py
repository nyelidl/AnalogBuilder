"""
plip_analyzer.py
================
PLIP-based interaction analysis for Analog Designer.

Functions
---------
run_plip_on_complex()     Run PLIP on a protein-ligand complex PDB.
parse_plip_to_table()     Parse PLIP XML → interaction table DataFrame.
plip_to_residue_tags()    Map PLIP interactions → AA tags for pocket_reference.
plip_cifp_vector()        Convert PLIP features → binary cIFP vector.
compare_cifp()            Tanimoto + retained/new/lost interaction analysis.
plip_to_void_residues()   Extract residues NOT contacted → void candidates.
unified_recommendation()  Combine PLIP + void → fragment recommendations.
"""

from __future__ import annotations

import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
# ── Auto-patch PLIP supplemental.py for numpy≥2 compatibility ────────────────
def _patch_plip_supplemental():
    """Fix OverflowError: np.uint32(negative) on numpy≥2 in PLIP 2.3.x."""
    try:
        import plip.basic.supplemental as _sup
        import inspect as _insp
        _src = _insp.getsourcefile(_sup)
        if _src is None:
            return
        with open(_src) as _f:
            _txt = _f.read()
        _old = (
            "    dct = {}\n"
            "    if int32 == 4294967295:  # Special case in some structures (note, this is just a workaround)\n"
            "        return -1\n"
            "    for i in range(-1000, -1):\n"
            "        dct[np.uint32(i)] = i\n"
            "    if int32 in dct:\n"
            "        return dct[int32]\n"
            "    else:\n"
            "        return int32"
        )
        _new = (
            "    if int32 == 4294967295:\n"
            "        return -1\n"
            "    try:\n"
            "        if int32 >= 2**31:\n"
            "            return int(int32) - 2**32\n"
            "        return int(int32)\n"
            "    except Exception:\n"
            "        return int(int32)"
        )
        if _old in _txt:
            with open(_src, 'w') as _f:
                _f.write(_txt.replace(_old, _new))
            # Reload the module
            import importlib as _imp
            _imp.reload(_sup)
    except Exception:
        pass

_patch_plip_supplemental()
# ─────────────────────────────────────────────────────────────────────────────



import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# PLIP interaction type → pocket property tag mapping
# ---------------------------------------------------------------------------

PLIP_TYPE_TO_TAGS: Dict[str, List[str]] = {
    "HBOND":       ["polar_hbond", "hbond_donor", "hbond_acceptor"],
    "WATERBRIDGE": ["polar_hbond", "hbond_acceptor"],
    "HYDROPHOBIC": ["hydrophobic"],
    "PISTACK":     ["aromatic", "hydrophobic"],
    "PICATION":    ["aromatic", "basic_positive"],
    "SALTBRIDGE":  ["acidic_negative", "basic_positive"],
    "HALOGEN":     ["sulfur_polarizable", "hbond_acceptor"],
    "METAL":       ["acidic_negative"],
}

# Which interaction types are "key" (losing them is a warning)
KEY_INTERACTION_TYPES = {"HBOND", "SALTBRIDGE", "METAL", "PICATION"}

# PLIP type → ligand design strategy
PLIP_DESIGN_STRATEGY: Dict[str, str] = {
    "HBOND":       "Preserve H-bond; add complementary donor/acceptor to extend",
    "HYDROPHOBIC": "Maintain hydrophobic contact; grow into adjacent hydrophobic space",
    "PISTACK":     "Keep π-stacking; add fused ring or halogen to strengthen",
    "PICATION":    "Preserve cation-π; keep basic N near aromatic residue",
    "SALTBRIDGE":  "Critical ionic contact — do NOT remove charged group",
    "HALOGEN":     "Halogen bond present; F/Cl on aromatic may strengthen",
    "WATERBRIDGE": "Water-mediated H-bond; direct H-bond replacement may improve",
    "METAL":       "Metal coordination — chelating group essential",
}


# ---------------------------------------------------------------------------
# Run PLIP
# ---------------------------------------------------------------------------

def run_plip_on_complex(
    complex_pdb: str,
    out_dir: str,
    name: str = "ligand",
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Run PLIP on a complex PDB file.

    Returns
    -------
    (xml_path, error_msg, log_text)
    xml_path is None if PLIP failed or is not installed.
    """
    plip_exe = shutil.which("plipcmd") or shutil.which("plip")
    if plip_exe is None:
        return None, "PLIP not installed (pip install plip)", ""

    od = Path(out_dir) / name
    od.mkdir(parents=True, exist_ok=True)

    import subprocess
    cmd = [plip_exe, "-f", str(complex_pdb), "-x", "-o", str(od)]
    res = subprocess.run(
        cmd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    log = res.stdout or ""
    (od / "plip_log.txt").write_text(log)

    xmls = sorted(od.glob("*.xml")) + sorted(od.glob("**/*.xml"))
    if res.returncode != 0 or not xmls:
        return None, f"PLIP failed (exit {res.returncode})", log

    return str(xmls[0]), None, log


# ---------------------------------------------------------------------------
# Parse PLIP XML → interaction table
# ---------------------------------------------------------------------------

def parse_plip_to_table(xml_path: str) -> pd.DataFrame:
    """
    Parse PLIP XML → DataFrame with one row per interaction.

    Columns: type, chain, resname, resnum, distance_A, ligand_atom, protein_atom
    """
    if not xml_path or not os.path.exists(xml_path):
        return pd.DataFrame()

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return pd.DataFrame()

    type_map = {
        "hydrophobic_interaction": "HYDROPHOBIC",
        "hydrogen_bond":           "HBOND",
        "water_bridge":            "WATERBRIDGE",
        "salt_bridge":             "SALTBRIDGE",
        "pi_stack":                "PISTACK",
        "pi_cation_interaction":   "PICATION",
        "halogen_bond":            "HALOGEN",
        "metal_complex":           "METAL",
    }

    rows = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        itype = next((v for k, v in type_map.items() if k in tag), None)
        if itype is None:
            continue

        vals = {}
        for child in elem.iter():
            k = child.tag.split("}")[-1].lower()
            vals[k] = (child.text or "").strip()

        def _get(*keys):
            for k in keys:
                v = vals.get(k, "")
                if v:
                    return v
            return ""

        resnr    = _get("resnr", "resnum", "res_nr")
        restype  = _get("restype", "resname", "res_type")
        reschain = _get("reschain", "chain")
        dist     = _get("dist", "distance", "dist_h-a", "dist_d-a", "dist_a-w")
        lig_atom = _get("ligcarbonidx", "ligatom", "lig_atom")
        prot_atom= _get("protcarbonidx", "protatom", "prot_atom")

        try:
            dist_f = float(dist) if dist else None
        except ValueError:
            dist_f = None

        rows.append({
            "type":       itype,
            "chain":      reschain or "_",
            "resname":    restype or "UNK",
            "resnum":     resnr or "0",
            "residue":    f"{restype}{resnr}:{reschain}" if restype and resnr else "UNK",
            "distance_A": round(dist_f, 2) if dist_f else None,
            "is_key":     itype in KEY_INTERACTION_TYPES,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).drop_duplicates(subset=["type","chain","resname","resnum"])
    return df.sort_values(["is_key","type"], ascending=[False, True]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# PLIP → residue tags for pocket_reference
# ---------------------------------------------------------------------------

# Standard 3-letter to 1-letter AA code
AA3_TO_1 = {
    "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C",
    "GLN":"Q","GLU":"E","GLY":"G","HIS":"H","ILE":"I",
    "LEU":"L","LYS":"K","MET":"M","MET":"M","PHE":"F",
    "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    "HID":"H","HIE":"H","HIP":"H","MSE":"M","SEC":"C",
}


def plip_to_residue_tags(
    plip_df: pd.DataFrame,
    aa_tags: Dict[str, List[str]],
) -> Tuple[Dict[str, int], List[str]]:
    """
    Convert PLIP interaction table → residue property tag counts.

    Returns
    -------
    tag_counts : Dict[tag → count]  (for pocket_reference.score_fragments)
    residue_codes : List[str]       (one-letter AA codes of contacted residues)
    """
    tag_counts: Dict[str, int] = {}
    residue_codes: List[str] = []

    for _, row in plip_df.iterrows():
        itype   = row["type"]
        resname = str(row["resname"]).upper()[:3]

        # Map interaction type → pocket tags
        for tag in PLIP_TYPE_TO_TAGS.get(itype, []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Map residue name → AA one-letter code → AA_TAGS
        aa1 = AA3_TO_1.get(resname, "")
        if aa1:
            residue_codes.append(aa1)
            for tag in aa_tags.get(aa1, []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    return tag_counts, list(set(residue_codes))


# ---------------------------------------------------------------------------
# cIFP vector
# ---------------------------------------------------------------------------

def build_cifp_vocabulary(all_features: List[List[str]]) -> List[str]:
    """Build sorted vocabulary from all feature lists."""
    vocab: Set[str] = set()
    for feats in all_features:
        vocab.update(feats)
    return sorted(vocab)


def cifp_vector(features: List[str], vocabulary: List[str]) -> np.ndarray:
    """Binary vector: 1 if feature in vocabulary."""
    feat_set = set(features)
    return np.array([1 if v in feat_set else 0 for v in vocabulary], dtype=np.float32)


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto similarity between two binary vectors."""
    inter = float(np.dot(a, b))
    union = float(np.sum(a) + np.sum(b) - inter)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Compare parent vs analogs
# ---------------------------------------------------------------------------

def compare_cifp(
    parent_features: List[str],
    analog_features_map: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Compare parent cIFP vs each analog cIFP.

    Parameters
    ----------
    parent_features    : PLIP features of parent (from parse_plip_xml)
    analog_features_map: {compound_name: [features]} for each analog

    Returns
    -------
    DataFrame with columns:
      compound, tanimoto, n_retained, n_lost, n_new,
      retained_key, lost_key, new_interactions, warning
    """
    all_feats = [parent_features] + list(analog_features_map.values())
    vocab = build_cifp_vocabulary(all_feats)

    parent_vec  = cifp_vector(parent_features, vocab)
    parent_set  = set(parent_features)
    parent_key  = {f for f in parent_features
                   if any(kt in f for kt in KEY_INTERACTION_TYPES)}

    rows = []
    for name, feats in analog_features_map.items():
        analog_vec = cifp_vector(feats, vocab)
        analog_set = set(feats)

        retained    = parent_set & analog_set
        lost        = parent_set - analog_set
        new_ints    = analog_set - parent_set
        lost_key    = parent_key & lost
        retained_key= parent_key & retained

        tan = tanimoto(parent_vec, analog_vec)

        # Warning if key interactions lost
        warning = ""
        if lost_key:
            warning = f"⚠️ Lost key: {', '.join(sorted(lost_key)[:3])}"

        rows.append({
            "compound":        name,
            "tanimoto":        round(tan, 3),
            "n_retained":      len(retained),
            "n_lost":          len(lost),
            "n_new":           len(new_ints),
            "retained_key":    len(retained_key),
            "lost_key":        len(lost_key),
            "new_interactions": "; ".join(sorted(new_ints)[:5]),
            "warning":         warning,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("tanimoto", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Void residues from PLIP (residues in pocket NOT contacted)
# ---------------------------------------------------------------------------

def plip_to_void_residues(
    plip_df: pd.DataFrame,
    all_pocket_residues: List[str],
) -> List[str]:
    """
    Return residue labels that are in the pocket but NOT in PLIP interactions.
    These define potential growth vectors.

    Parameters
    ----------
    plip_df             : from parse_plip_to_table()
    all_pocket_residues : residue_label list from analyze_complex_distance_shell()
    """
    contacted = set(plip_df["residue"].tolist()) if not plip_df.empty else set()
    return [r for r in all_pocket_residues if r not in contacted]


# ---------------------------------------------------------------------------
# Unified recommendation
# ---------------------------------------------------------------------------

def unified_recommendation(
    plip_df: pd.DataFrame,
    void_subpockets: List[Dict],
    plip_tag_counts: Dict[str, int],
    aa_tags: Dict[str, List[str]],
) -> Dict:
    """
    Combine PLIP + void analysis → unified design recommendation.

    Returns
    -------
    {
      "preserve":  [{"residue", "type", "strategy"}],   # interactions to keep
      "grow":      [{"sub_pocket_id", "size_class",
                     "void_volume", "aa_codes",
                     "strategy", "example_fragments"}],  # vectors to grow
      "summary":   str,
    }
    """
    # ── Interactions to preserve ─────────────────────────────────────────────
    preserve = []
    if not plip_df.empty:
        for _, row in plip_df[plip_df["is_key"]].iterrows():
            preserve.append({
                "residue":  row["residue"],
                "type":     row["type"],
                "strategy": PLIP_DESIGN_STRATEGY.get(row["type"], ""),
            })

    # ── Growth vectors from void analysis ────────────────────────────────────
    from void_analyzer import SIZE_EXAMPLES
    grow = []
    for sp in void_subpockets:
        aa_in_sp = sp.get("aa_codes", [])
        # Build local tag counts for this sub-pocket
        local_tags: Dict[str, int] = {}
        for aa in aa_in_sp:
            for tag in aa_tags.get(aa, []):
                local_tags[tag] = local_tags.get(tag, 0) + 1
        dominant = max(local_tags, key=local_tags.get) if local_tags else "hydrophobic"

        grow.append({
            "sub_pocket_id":   sp["sub_pocket_id"],
            "size_class":      sp["size_class"],
            "void_volume_A3":  sp["void_volume_est_A3"],
            "available_r_A":   sp["available_radius_A"],
            "residue_labels":  sp["residue_labels"][:4],
            "aa_codes":        aa_in_sp,
            "dominant_tag":    dominant,
            "strategy":        f"Fill {sp['size_class']} void (r≈{sp['available_radius_A']:.1f}Å) "
                               f"— {dominant.replace('_',' ')} environment",
            "example_frags":   SIZE_EXAMPLES.get(sp["size_class"], ""),
        })

    # ── Summary text ──────────────────────────────────────────────────────────
    n_key   = len(preserve)
    n_voids = len(grow)
    summary_parts = []
    if n_key:
        key_types = list({p["type"] for p in preserve})
        summary_parts.append(
            f"**{n_key} key interaction(s)** to preserve "
            f"({', '.join(key_types[:3])})"
        )
    if n_voids:
        sizes = [g["size_class"] for g in grow]
        summary_parts.append(
            f"**{n_voids} unoccupied sub-pocket(s)** detected "
            f"({', '.join(sizes)})"
        )
    if not summary_parts:
        summary_parts = ["No strong contacts detected — try relaxing PLIP cutoffs"]

    summary = "  ·  ".join(summary_parts)

    return {
        "preserve": preserve,
        "grow":     grow,
        "summary":  summary,
    }


# ---------------------------------------------------------------------------
# Convenience: fallback distance-based cIFP if PLIP unavailable
# ---------------------------------------------------------------------------

def distance_cifp_features(complex_pdb: str, cutoff: float = 4.0) -> List[str]:
    """
    Fallback when PLIP is not installed: distance-based contact fingerprint.
    Returns features in PLIP-compatible format: "CONTACT:chain:resname:resnum"
    """
    features: Set[str] = set()
    protein, ligand = [], []

    with open(complex_pdb, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            elem = line[76:78].strip().upper() or line[12:16].strip()[:1].upper()
            if elem == "H":
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except Exception:
                continue
            rec = {
                "name":    line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain":   line[21].strip() or "_",
                "resnum":  line[22:26].strip(),
                "x": x, "y": y, "z": z,
            }
            if line.startswith("ATOM"):
                protein.append(rec)
            else:
                ligand.append(rec)

    c2 = cutoff ** 2
    for pa in protein:
        for la in ligand:
            dx = pa["x"] - la["x"]
            dy = pa["y"] - la["y"]
            dz = pa["z"] - la["z"]
            if dx*dx + dy*dy + dz*dz <= c2:
                features.add(
                    f"CONTACT:{pa['chain']}:{pa['resname']}:{pa['resnum']}"
                )
                break

    return sorted(features)
