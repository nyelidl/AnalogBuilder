"""
fpocket_analyzer.py
===================
Run fpocket on a protein structure and parse results.
Identifies the pocket nearest to the co-crystal ligand (if available).

Dependencies:
  - fpocket binary (installed explicitly with scripts/install_fpocket.py or conda)
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
    configured = os.environ.get("FPOCKET_BIN")
    local = Path(__file__).parent / ".tools/fpocket/bin/fpocket"
    for candidate in (configured, str(local)):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return str(Path(candidate).resolve())
    for name in ("fpocket", "fpocket4", "fpocket3"):
        exe = shutil.which(name)
        if exe:
            return exe
    # Common manual install locations
    for path in ("/opt/homebrew/bin/fpocket", "/usr/local/bin/fpocket", "/opt/fpocket/bin/fpocket",
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
    exe = _find_fpocket()
    if not exe:
        return [], "", "fpocket unavailable; run python scripts/install_fpocket.py or conda install -c conda-forge fpocket"
    if not (0 < min_sphere < max_sphere):
        return [], "", "Alpha-sphere radii must satisfy 0 < min < max."
    source = Path(protein_pdb)
    if not source.is_file():
        return [], "", f"Protein PDB not found: {source}"
    if source.suffix.lower() != ".pdb":
        return [], "", "fpocket requires PDB coordinates; select a protein .pdb, not PDBQT/mmCIF."
    # Keep each run's results in its own directory: returned paths stay valid,
    # and a failed run cannot pick up output from an earlier run.
    root = Path(out_dir).resolve() if out_dir else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="fpocket_", dir=root))
    pdb_copy = run_dir / "protein.pdb"
    protein_lines = []
    for line in source.read_text(errors="replace").splitlines(True):
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM  ", "TER   ")):
            protein_lines.append(line)
    if not any(line.startswith("ATOM  ") for line in protein_lines):
        return [], "", "No protein ATOM records found in input PDB."
    pdb_copy.write_text("".join(protein_lines) + "END\n")
    cmd = [exe, "-f", str(pdb_copy), "-m", str(min_sphere), "-M", str(max_sphere)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=run_dir)
        log = (res.stdout or "") + (res.stderr or "")
    except subprocess.TimeoutExpired:
        return [], "", f"fpocket timed out after {timeout}s"
    except OSError as exc:
        return [], "", f"Could not start fpocket: {exc}"
    (run_dir / "fpocket_log.txt").write_text(log)
    if res.returncode:
        return [], log, f"fpocket exited with status {res.returncode}; inspect the run log."
    fp_out = run_dir / "protein_out"
    if not fp_out.is_dir():
        return [], log, "fpocket produced no output directory"
    pockets = _parse_fpocket_output(str(fp_out))
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

        # fpocket 4 writes alpha spheres as PQR, not *_vert.pdb.
        vert_pdb = next((str(atm_pdb.with_name(atm_pdb.name.replace("_atm.pdb", suffix)))
                         for suffix in ("_vert.pqr", "_vert.pdb")
                         if atm_pdb.with_name(atm_pdb.name.replace("_atm.pdb", suffix)).exists()), None)
        spheres = _parse_pocket_atm_pdb(vert_pdb) if vert_pdb else []
        points = spheres or residues
        if not points:
            continue
        centroid = np.mean([[r["x"], r["y"], r["z"]] for r in points], axis=0).tolist()

        pocket = {
            "pocket_id":          pid,
            "centroid_xyz":       centroid,
            "n_residues":         len(set(f"{r['chain']}{r['resnum']}" for r in residues)),
            "residue_atoms":      residues,
            "sphere_atoms":       spheres,
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
                    "chain":     line[21].strip() or "_",
                    "resnum":    line[22:26].strip() + line[26:27].strip(),
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

    Return None if no pocket is within max_dist_A; never silently select a remote pocket.

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
    return None


# ── Build residue DataFrame for selected pocket ───────────────────────────────
def pocket_residues_df(pocket: Dict, ligand_atoms=None, contact_cutoff=4.0) -> pd.DataFrame:
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
        res_atoms = [r for r in atoms if (r["chain"], r["resnum"], r["resname"]) == (chain, resnum, resname)]
        distance = None
        if ligand_atoms:
            xyz = np.array([[r["x"], r["y"], r["z"]] for r in res_atoms])
            lig = np.array([r["xyz"] for r in ligand_atoms])
            distance = float(np.linalg.norm(xyz[:, None, :] - lig[None, :, :], axis=2).min())
        contacted = distance is not None and distance <= contact_cutoff
        rows.append({
            "resname":         resname,
            "chain":           chain,
            "resnum":          resnum,
            "residue_label":   f"{resname}{resnum}:{chain}",
            "aa_one":          one,
            "property_tags":   ",".join(AA_TAGS.get(one, [])) if one else "",
            "min_dist_to_ligand_A": distance,
            "is_pocket_residue":   True,
            "is_contacted":        contacted,
            "is_noncontact_growth_residue": distance is not None and not contacted,
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

    res_df = pocket_residues_df(selected, ligand_atoms)
    return res_df, pockets, selected, log


# Compatibility helper: check only; installation is an explicit setup action.
def ensure_fpocket_installed() -> Tuple[bool, str]:
    exe = _find_fpocket()
    return bool(exe), (f"fpocket available: {exe}" if exe else
                       "Run python scripts/install_fpocket.py or conda install -c conda-forge fpocket")
