"""
fpocket_analyzer.py
===================
Run fpocket on a protein structure and parse results.
Identifies the pocket nearest to the co-crystal ligand (if available).

Dependencies:
  - fpocket binary (auto-installed by setup or via apt/conda)
  - numpy, pandas (always available)

Main entry points:
  run_fpocket()          → run fpocket on protein PDB, parse all pockets
  select_ligand_pocket() → pick the pocket closest to the co-crystal ligand
  pocket_residues_df()   → return residue DataFrame for selected pocket
"""

from __future__ import annotations
import os, re, shutil, subprocess, tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── Find fpocket binary ───────────────────────────────────────────────────────
def _find_fpocket() -> Optional[str]:
    for name in ("fpocket", "fpocket4", "fpocket3"):
        exe = shutil.which(name)
        if exe:
            return exe
    # Common manual install locations
    for path in ("/usr/local/bin/fpocket", "/opt/fpocket/bin/fpocket",
                 str(Path.home() / ".local/bin/fpocket")):
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None

FPOCKET_EXE = _find_fpocket()
FPOCKET_OK  = FPOCKET_EXE is not None


# ── Run fpocket ───────────────────────────────────────────────────────────────
def run_fpocket(
    protein_pdb: str,
    out_dir:     Optional[str] = None,
    min_sphere:  int   = 3,
    max_sphere:  int   = 6,
    timeout:     int   = 60,
) -> Tuple[List[Dict], str, str]:
    """
    Run fpocket on a protein PDB and return parsed pocket list.

    Parameters
    ----------
    protein_pdb : path to protein-only PDB (no ligand — fpocket works on apo)
    out_dir     : where to write fpocket output (default: temp dir)
    min_sphere  : min alpha-sphere radius Å (default 3)
    max_sphere  : max alpha-sphere radius Å (default 6)

    Returns
    -------
    (pockets, log, error)
    pockets: list of dicts, one per fpocket pocket, sorted by druggability score desc
    log:     raw fpocket stdout
    error:   error message string (empty if OK)
    """
    if not FPOCKET_OK:
        return [], "", "fpocket not installed. Install with: sudo apt install fpocket"

    protein_pdb = str(protein_pdb)
    if not os.path.exists(protein_pdb):
        return [], "", f"Protein PDB not found: {protein_pdb}"

    # fpocket writes output next to the input file — use a temp copy
    tmp_dir  = tempfile.mkdtemp()
    pdb_copy = os.path.join(tmp_dir, "protein.pdb")
    shutil.copy2(protein_pdb, pdb_copy)

    cmd = [
        FPOCKET_EXE,
        "-f", pdb_copy,
        "-m", str(min_sphere),
        "-M", str(max_sphere),
    ]

    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp_dir
        )
        log = res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        return [], "", f"fpocket timed out after {timeout}s"
    except Exception as e:
        return [], "", f"fpocket error: {e}"

    # fpocket writes to protein_out/ relative to the PDB location
    fp_out = os.path.join(tmp_dir, "protein_out")
    if not os.path.isdir(fp_out):
        return [], log, "fpocket produced no output directory"

    pockets = _parse_fpocket_output(fp_out)

    # Copy to user-specified out_dir if provided
    if out_dir:
        import shutil as _sh
        os.makedirs(out_dir, exist_ok=True)
        _sh.copytree(fp_out, os.path.join(out_dir, "fpocket_out"),
                     dirs_exist_ok=True)
        # Write pocket PDB paths relative to out_dir
        for p in pockets:
            for k in ("atm_pdb", "vert_pdb"):
                if p.get(k):
                    rel = os.path.relpath(p[k], tmp_dir)
                    p[f"{k}_out"] = os.path.join(out_dir, "fpocket_out",
                                                   os.path.basename(p[k]))

    return pockets, log, ""


# ── Parse fpocket output ──────────────────────────────────────────────────────
def _parse_fpocket_output(fp_out: str) -> List[Dict]:
    """Parse fpocket output directory into a list of pocket dicts."""
    pockets = []

    # Parse info.txt for scores
    info_file = os.path.join(fp_out, "protein_info.txt")
    scores_by_id: Dict[int, Dict] = {}
    if os.path.exists(info_file):
        scores_by_id = _parse_fpocket_info(info_file)

    # Find per-pocket residue PDBs
    pockets_dir = os.path.join(fp_out, "pockets")
    if not os.path.isdir(pockets_dir):
        return []

    atm_pdbs = sorted(Path(pockets_dir).glob("pocket*_atm.pdb"))
    for atm_pdb in atm_pdbs:
        m = re.search(r"pocket(\d+)_atm", atm_pdb.name)
        if not m:
            continue
        pid = int(m.group(1))
        residues = _parse_pocket_atm_pdb(str(atm_pdb))

        # Get centroid from residue atoms
        if residues:
            xyz = np.array([[r["x"], r["y"], r["z"]] for r in residues])
            centroid = xyz.mean(axis=0).tolist()
        else:
            centroid = [0.0, 0.0, 0.0]

        # Alpha-sphere PDB (pocket vertices)
        vert_pdb = str(atm_pdb).replace("_atm.pdb", "_vert.pdb")

        pocket = {
            "pocket_id":          pid,
            "centroid_xyz":       centroid,
            "n_residues":         len(set(f"{r['chain']}{r['resnum']}" for r in residues)),
            "residue_atoms":      residues,
            "atm_pdb":            str(atm_pdb),
            "vert_pdb":           vert_pdb if os.path.exists(vert_pdb) else None,
            **scores_by_id.get(pid, {}),
        }
        pockets.append(pocket)

    # Sort by druggability score (fpocket ranks by volume × other factors)
    pockets.sort(key=lambda p: (
        -float(p.get("druggability_score", 0) or 0)
    ))
    return pockets


def _parse_fpocket_info(info_file: str) -> Dict[int, Dict]:
    """Parse fpocket *_info.txt to extract per-pocket scores."""
    scores: Dict[int, Dict] = {}
    current_id = None
    current = {}

    with open(info_file, errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"Pocket\s+(\d+)\s*:", line, re.IGNORECASE)
            if m:
                if current_id is not None:
                    scores[current_id] = current
                current_id = int(m.group(1))
                current = {}
                continue
            # Key-value pairs like "Score :                  19.6838"
            kv = re.match(r"(.+?)\s*:\s*(-?[\d.]+)", line)
            if kv and current_id is not None:
                key = kv.group(1).strip().lower().replace(" ", "_").replace("-","_")
                try:
                    current[key] = float(kv.group(2))
                except ValueError:
                    pass

    if current_id is not None:
        scores[current_id] = current

    # Normalise common key names
    normalized: Dict[int, Dict] = {}
    for pid, raw in scores.items():
        normalized[pid] = {
            "fpocket_score":       raw.get("score", raw.get("pocket_score", 0)),
            "druggability_score":  raw.get("druggability_score", raw.get("drug._score", 0)),
            "volume_A3":           raw.get("volume", raw.get("volume_(a^3)", 0)),
            "mean_local_hydrophobic_density": raw.get("mean_local_hydrophobic_density", 0),
            "mean_alpha_sphere_radius": raw.get("mean_alpha_sphere_radius", 0),
            "n_alpha_spheres":     raw.get("number_of_alpha_spheres", 0),
        }
    return normalized


def _parse_pocket_atm_pdb(atm_pdb: str) -> List[Dict]:
    """Parse pocket*_atm.pdb to get residue atoms with XYZ."""
    atoms = []
    with open(atm_pdb, errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                atoms.append({
                    "atom_name": line[12:16].strip(),
                    "resname":   line[17:20].strip(),
                    "chain":     line[21].strip() or "A",
                    "resnum":    line[22:26].strip(),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except Exception:
                pass
    return atoms


# ── Select pocket closest to ligand ──────────────────────────────────────────
def select_ligand_pocket(
    pockets:      List[Dict],
    ligand_atoms: List[Dict],  # from core.read_complex_atoms_for_pocket
    max_dist_A:   float = 8.0,
) -> Optional[Dict]:
    """
    From a list of fpocket pockets, return the one whose centroid is
    closest to the co-crystal ligand centroid.

    Falls back to the top-scored pocket if none within max_dist_A.

    Parameters
    ----------
    pockets      : from run_fpocket()
    ligand_atoms : list of dicts with 'xyz' key (numpy array)
    max_dist_A   : max centroid-to-ligand-centroid distance to consider

    Returns
    -------
    The selected pocket dict, or None if no pockets.
    """
    if not pockets:
        return None
    if not ligand_atoms:
        return pockets[0]  # fall back to top-scored

    lig_centroid = np.mean(
        [a["xyz"] for a in ligand_atoms], axis=0
    )

    best = None
    best_dist = float("inf")
    for p in pockets:
        c = np.array(p["centroid_xyz"])
        d = float(np.linalg.norm(c - lig_centroid))
        p["dist_to_ligand_centroid_A"] = round(d, 2)
        if d < best_dist:
            best_dist = d
            best = p

    if best_dist <= max_dist_A:
        return best
    # None within threshold — return top scored
    return pockets[0]


# ── Build residue DataFrame for selected pocket ───────────────────────────────
def pocket_residues_df(pocket: Dict) -> pd.DataFrame:
    """
    Convert fpocket pocket dict to a residue-level DataFrame
    compatible with the rest of Analog Designer's pocket pipeline.

    Columns match analyze_complex_distance_shell() output:
      resname, chain, resnum, residue_label, aa_one, property_tags
    """
    atoms = pocket.get("residue_atoms", [])
    if not atoms:
        return pd.DataFrame()

    # Deduplicate by residue identity
    seen = {}
    for a in atoms:
        key = (a["chain"], a["resnum"], a["resname"])
        if key not in seen:
            seen[key] = a

    # Import AA_TAGS from core if available
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from core import AA_TAGS, AA3_TO_ONE
    except ImportError:
        AA_TAGS    = {}
        AA3_TO_ONE = {}

    rows = []
    for (chain, resnum, resname), a in seen.items():
        one = AA3_TO_ONE.get(resname, "")
        rows.append({
            "resname":         resname,
            "chain":           chain,
            "resnum":          resnum,
            "residue_label":   f"{resname}{resnum}:{chain}",
            "aa_one":          one,
            "property_tags":   ",".join(AA_TAGS.get(one, [])) if one else "",
            "min_dist_to_ligand_A": pocket.get("dist_to_ligand_centroid_A", 0.0),
            "is_pocket_residue":   True,
            "is_contacted":        False,  # fpocket doesn't classify this
            "is_noncontact_growth_residue": True,
        })

    return pd.DataFrame(rows)


# ── Convenience: full pipeline ────────────────────────────────────────────────
def fpocket_pocket_analysis(
    protein_pdb:   str,
    ligand_atoms:  Optional[List[Dict]] = None,
    work_dir:      Optional[str] = None,
    min_sphere:    int   = 3,
    max_sphere:    int   = 6,
) -> Tuple[Optional[pd.DataFrame], List[Dict], Dict, str]:
    """
    Full fpocket pipeline:
      1. Run fpocket on protein PDB
      2. Select pocket nearest to ligand (or top-scored if no ligand)
      3. Return residue DataFrame + all pockets + selected pocket + log

    Returns
    -------
    (residue_df, all_pockets, selected_pocket, log_or_error)
    """
    pockets, log, error = run_fpocket(
        protein_pdb, out_dir=work_dir,
        min_sphere=min_sphere, max_sphere=max_sphere,
    )
    if error:
        return None, [], {}, error
    if not pockets:
        return None, [], {}, "fpocket found no pockets in this structure."

    selected = select_ligand_pocket(pockets, ligand_atoms or [])
    if selected is None:
        return None, pockets, {}, "Could not select a pocket near the ligand."

    res_df = pocket_residues_df(selected)
    return res_df, pockets, selected, log


# ── Install helper ─────────────────────────────────────────────────────────────
def ensure_fpocket_installed() -> Tuple[bool, str]:
    """
    Check if fpocket is installed; try to install if not.
    Returns (success, message).
    """
    global FPOCKET_EXE, FPOCKET_OK
    if FPOCKET_OK:
        return True, f"fpocket available: {FPOCKET_EXE}"

    import subprocess as _sp
    # Try apt
    r = _sp.run(["apt-get", "install", "-y", "-q", "fpocket"],
                capture_output=True, text=True)
    exe = _find_fpocket()
    if exe:
        FPOCKET_EXE = exe
        FPOCKET_OK  = True
        return True, f"fpocket installed via apt: {exe}"

    return False, (
        "fpocket not found. Install with:\n"
        "  Ubuntu/Debian: sudo apt install fpocket\n"
        "  macOS:         brew install fpocket\n"
        "  Build source:  https://github.com/Discngine/fpocket"
    )
