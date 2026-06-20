"""
void_analyzer.py
================
Pocket void / unoccupied space analysis for Analog Designer.

Works in two modes:
  A. Co-crystal complex (PDB with ligand) — measure void from 3D coordinates
  B. Protein-only (no ligand) — estimate void from pocket geometry

Output per attachment vector:
  - sub_pocket_id     : which spatial cluster the growth zone belongs to
  - available_radius  : estimated available space (Å) for a new fragment
  - size_class        : small / medium / large / extended
  - recommended_frags : fragment names that fit the space
  - residue_labels    : pocket residues bounding this sub-pocket
  - void_volume_est   : rough sphere volume estimate (ų)

No external tools required — pure numpy + RDKit geometry.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# vdW radii (Å) for common heavy atoms
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80,
       "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98,
       "P": 1.80, "default": 1.70}

# Fragment size class → estimated 3D radius of the group (Å from attachment bond)
# Based on maximum extension of group from attachment point
SIZE_RADIUS_A = {
    "small":    1.8,   # F, Cl, OH, Me, CN
    "medium":   2.8,   # cyclopropyl, OMe, CF3, azetidine
    "large":    4.0,   # phenyl, piperidine, pyridine
    "extended": 5.5,   # indole, biphenyl, N-methylpiperazine benzyl
}

# Bond length from parent atom to first fragment atom (Å)
ATTACH_BOND_LEN = 1.52   # C-C bond; covers most cases

# Probe radius for "available space" calculation (Å) — ligand atom vdW
LIGAND_PROBE_R = 1.70


# ---------------------------------------------------------------------------
# Mode A: co-crystal complex void analysis
# ---------------------------------------------------------------------------

def _atom_vdw(atom_name: str, element: str = "") -> float:
    el = element.upper() or atom_name.strip()[:1].upper()
    return VDW.get(el, VDW["default"])


def _growth_residue_xyz(protein_atoms: List[Dict], resnum: str,
                        chain: str) -> Optional[np.ndarray]:
    """
    Return centroid XYZ of all heavy atoms in a given residue.
    Used to place sub-pocket cluster centroid.
    """
    pts = [a["xyz"] for a in protein_atoms
           if a["resnum"] == resnum and a["chain"] == chain]
    return np.mean(pts, axis=0) if pts else None


def _ligand_centroid(ligand_atoms: List[Dict]) -> np.ndarray:
    return np.mean([a["xyz"] for a in ligand_atoms], axis=0)


def _ligand_surface_point_toward(lig_atoms: List[Dict],
                                 direction: np.ndarray) -> np.ndarray:
    """
    Find the ligand atom furthest in `direction` — approximates
    the ligand surface edge closest to a growth residue.
    """
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    lig_xyz = np.array([a["xyz"] for a in lig_atoms])
    proj = lig_xyz @ direction
    idx = int(np.argmax(proj))
    return lig_xyz[idx] + direction * LIGAND_PROBE_R


def compute_void_from_complex(
    protein_atoms: List[Dict],
    ligand_atoms:  List[Dict],
    growth_df:     pd.DataFrame,
    pocket_df:     pd.DataFrame,
    contact_cutoff: float = 4.0,
    pocket_cutoff:  float = 6.0,
    cluster_dist:   float = 6.0,
) -> List[Dict]:
    """
    Mode A: co-crystal complex.

    For each non-contact growth residue, estimate the available void
    between the ligand surface and the residue atoms.

    Returns list of sub-pocket dicts.
    """
    if growth_df.empty:
        return []

    lig_xyz = np.array([a["xyz"] for a in ligand_atoms])
    lig_centroid = lig_xyz.mean(axis=0)

    # Build residue centroid map
    res_centroids: Dict[Tuple, np.ndarray] = {}
    for a in protein_atoms:
        key = (a["chain"], a["resnum"])
        if key not in res_centroids:
            res_centroids[key] = []
        res_centroids[key].append(a["xyz"])
    res_centroids = {k: np.mean(v, axis=0) for k, v in res_centroids.items()}

    # ── Per-residue void estimate ─────────────────────────────────────────
    res_data = []
    for _, row in growth_df.iterrows():
        key = (row["chain"], row["resnum"])
        res_center = res_centroids.get(key)
        if res_center is None:
            continue

        # Direction from ligand centroid to residue center
        direction = res_center - lig_centroid
        dist_to_center = float(np.linalg.norm(direction))
        if dist_to_center < 1e-3:
            continue

        # Ligand surface point toward this residue
        lig_edge = _ligand_surface_point_toward(ligand_atoms, direction)

        # Distance from ligand edge to residue centroid
        gap_vec  = res_center - lig_edge
        gap_dist = float(np.linalg.norm(gap_vec))

        # Available radius = half the gap (fragment fills from ligand edge)
        available_r = max(0.5, gap_dist * 0.55)

        res_data.append({
            "resnum":      row["resnum"],
            "chain":       row["chain"],
            "resname":     row["resname"],
            "residue_label": row.get("residue_label", f"{row['resname']}{row['resnum']}"),
            "aa_one":      row.get("aa_one", ""),
            "dmin":        float(row["min_dist_to_ligand_A"]),
            "gap_dist":    round(gap_dist, 2),
            "available_r": round(available_r, 2),
            "center":      res_center,
        })

    if not res_data:
        return []

    # ── Cluster into sub-pockets ──────────────────────────────────────────
    return _cluster_into_subpockets(res_data, cluster_dist)


# ---------------------------------------------------------------------------
# Mode B: protein-only (no co-crystal ligand)
# ---------------------------------------------------------------------------

def compute_void_from_protein_only(
    pocket_df:     pd.DataFrame,
    protein_atoms: List[Dict],
    pocket_cutoff: float = 6.0,
    contact_cutoff: float = 4.0,
    cluster_dist:   float = 6.0,
) -> List[Dict]:
    """
    Mode B: protein without co-crystal ligand.

    Estimates pocket void from residue geometry alone:
    - Residues within pocket_cutoff form the pocket boundary
    - Void volume estimated as sphere fitting between residue centroids
    - Uses convex hull diameter of pocket residues as pocket size proxy

    Returns list of sub-pocket dicts (less precise than Mode A).
    """
    if pocket_df.empty:
        return []

    # Build residue centroid map
    res_centroids: Dict[Tuple, np.ndarray] = {}
    for a in protein_atoms:
        key = (a["chain"], a["resnum"])
        if key not in res_centroids:
            res_centroids[key] = []
        res_centroids[key].append(a["xyz"])
    res_centroids = {k: np.mean(v, axis=0) for k, v in res_centroids.items()}

    # Pocket centroid
    centers = []
    for _, row in pocket_df.iterrows():
        key = (row["chain"], row["resnum"])
        if key in res_centroids:
            centers.append(res_centroids[key])
    if not centers:
        return []
    pocket_centroid = np.mean(centers, axis=0)

    # All residues = potential growth zones (no ligand to exclude)
    res_data = []
    for _, row in pocket_df.iterrows():
        key = (row["chain"], row["resnum"])
        res_center = res_centroids.get(key)
        if res_center is None:
            continue

        # Distance from pocket centroid to residue
        d_from_center = float(np.linalg.norm(res_center - pocket_centroid))

        # Available radius estimate:
        # Inner residues (close to centroid) → more available space
        # Outer residues → tighter, less space
        dmin = float(row.get("min_dist_to_ligand_A", d_from_center))
        # Use distance from center scaled to pocket size
        available_r = max(0.5, (pocket_cutoff - dmin) * 0.6 + 1.0)

        res_data.append({
            "resnum":      row["resnum"],
            "chain":       row["chain"],
            "resname":     row["resname"],
            "residue_label": row.get("residue_label", f"{row['resname']}{row['resnum']}"),
            "aa_one":      row.get("aa_one", ""),
            "dmin":        dmin,
            "gap_dist":    round(d_from_center, 2),
            "available_r": round(available_r, 2),
            "center":      res_center,
        })

    if not res_data:
        return []

    return _cluster_into_subpockets(res_data, cluster_dist)


# ---------------------------------------------------------------------------
# Clustering + size classification (shared)
# ---------------------------------------------------------------------------

def _cluster_into_subpockets(
    res_data: List[Dict],
    cluster_dist: float = 6.0,
) -> List[Dict]:
    """
    Single-linkage clustering by Euclidean distance between residue centroids.
    Returns one dict per sub-pocket.
    """
    n = len(res_data)
    if n == 0:
        return []

    centers = np.array([r["center"] for r in res_data])
    assigned = [-1] * n
    cluster_id = 0

    for i in range(n):
        if assigned[i] >= 0:
            continue
        assigned[i] = cluster_id
        changed = True
        while changed:
            changed = False
            for j in range(n):
                if assigned[j] >= 0:
                    continue
                # Check distance to any member of current cluster
                for k in range(n):
                    if assigned[k] == cluster_id:
                        if np.linalg.norm(centers[k] - centers[j]) <= cluster_dist:
                            assigned[j] = cluster_id
                            changed = True
                            break
        cluster_id += 1

    # Build sub-pocket summaries
    clusters: Dict[int, List[Dict]] = {}
    for i, cid in enumerate(assigned):
        clusters.setdefault(cid, []).append(res_data[i])

    subpockets = []
    for cid, members in sorted(clusters.items()):
        avg_r   = float(np.mean([m["available_r"] for m in members]))
        max_r   = float(np.max([m["available_r"] for m in members]))
        avg_gap = float(np.mean([m["gap_dist"] for m in members]))

        size_class = _radius_to_size_class(avg_r)
        vol_est    = (4/3) * math.pi * avg_r ** 3

        centroid = np.mean([m["center"] for m in members], axis=0)

        subpockets.append({
            "sub_pocket_id":     cid + 1,
            "n_residues":        len(members),
            "residue_labels":    [m["residue_label"] for m in members],
            "aa_codes":          [m["aa_one"] for m in members if m["aa_one"]],
            "avg_gap_dist_A":    round(avg_gap, 2),
            "available_radius_A":round(avg_r, 2),
            "max_radius_A":      round(max_r, 2),
            "size_class":        size_class,
            "void_volume_est_A3":round(vol_est, 1),
            "centroid_xyz":      centroid.tolist(),
            "members":           members,
        })

    # Sort by available radius descending (biggest space first)
    subpockets.sort(key=lambda x: -x["available_radius_A"])
    return subpockets


def _radius_to_size_class(r: float) -> str:
    if r < 2.0:   return "small"
    if r < 3.2:   return "medium"
    if r < 4.5:   return "large"
    return "extended"


# ---------------------------------------------------------------------------
# Fragment filtering by size
# ---------------------------------------------------------------------------

def filter_frags_by_size(frags, size_class: str, allow_smaller: bool = True):
    """
    Filter fragment library to those that fit within a given size class.
    allow_smaller=True (default): also include smaller fragments.
    """
    order = ["small", "medium", "large", "extended"]
    max_idx = order.index(size_class)

    def _fits(f) -> bool:
        fsc = getattr(f, "size_class", "auto")
        if fsc == "auto":
            # Infer from heavy atom count
            from rdkit import Chem
            m = Chem.MolFromSmiles(f.smiles.replace("[*]", "[H]"))
            if m is None:
                return False
            n = m.GetNumHeavyAtoms()
            if n <= 2:   fsc = "small"
            elif n <= 5: fsc = "medium"
            elif n <= 10:fsc = "large"
            else:        fsc = "extended"
        try:
            idx = order.index(fsc)
        except ValueError:
            return True
        return idx <= max_idx if allow_smaller else idx == max_idx

    return [f for f in frags if _fits(f)]


# ---------------------------------------------------------------------------
# Main entry point: auto-detect mode
# ---------------------------------------------------------------------------

def analyze_void(
    protein_atoms: List[Dict],
    ligand_atoms:  Optional[List[Dict]],
    pocket_df:     pd.DataFrame,
    growth_df:     pd.DataFrame,
    contact_cutoff: float = 4.0,
    pocket_cutoff:  float = 6.0,
    cluster_dist:   float = 6.0,
) -> Tuple[List[Dict], str]:
    """
    Auto-detect Mode A vs Mode B and return sub-pocket list + mode string.

    Parameters
    ----------
    protein_atoms : from core.read_complex_atoms_for_pocket()
    ligand_atoms  : from core.read_complex_atoms_for_pocket() — None if no ligand
    pocket_df     : from core.analyze_complex_distance_shell()
    growth_df     : non-contact growth residues (idem)

    Returns
    -------
    (subpockets, mode_str)
    """
    has_ligand = ligand_atoms is not None and len(ligand_atoms) > 0

    if has_ligand and not growth_df.empty:
        mode = "co-crystal"
        subpockets = compute_void_from_complex(
            protein_atoms, ligand_atoms, growth_df, pocket_df,
            contact_cutoff=contact_cutoff,
            pocket_cutoff=pocket_cutoff,
            cluster_dist=cluster_dist,
        )
    else:
        mode = "protein-only"
        subpockets = compute_void_from_protein_only(
            pocket_df, protein_atoms,
            pocket_cutoff=pocket_cutoff,
            contact_cutoff=contact_cutoff,
            cluster_dist=cluster_dist,
        )

    return subpockets, mode


# ---------------------------------------------------------------------------
# Summary DataFrame (for Streamlit display)
# ---------------------------------------------------------------------------

def subpockets_to_df(subpockets: List[Dict]) -> pd.DataFrame:
    """Convert sub-pocket list to display DataFrame."""
    if not subpockets:
        return pd.DataFrame()
    rows = []
    for sp in subpockets:
        rows.append({
            "Sub-pocket":      sp["sub_pocket_id"],
            "Residues":        ", ".join(sp["residue_labels"][:5]) +
                               (f" +{len(sp['residue_labels'])-5}" if len(sp['residue_labels']) > 5 else ""),
            "Gap dist (Å)":    sp["avg_gap_dist_A"],
            "Available r (Å)": sp["available_radius_A"],
            "Size class":      sp["size_class"],
            "Est. volume (ų)": sp["void_volume_est_A3"],
            "Fit fragment":    SIZE_RADIUS_A.get(sp["size_class"], "?"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Recommendation text
# ---------------------------------------------------------------------------

SIZE_EXAMPLES = {
    "small":    "F, Cl, OH, CH₃, CN",
    "medium":   "cyclopropyl, OMe, CF₃, azetidine",
    "large":    "phenyl, piperidine, pyridine, oxetane",
    "extended": "indole, biphenyl, benzimidazole, N-methylpiperazine-benzyl",
}

def recommend_fragments_for_subpocket(sp: Dict, frag_library) -> List:
    """Return filtered + size-matched fragment list for one sub-pocket."""
    return filter_frags_by_size(frag_library, sp["size_class"], allow_smaller=True)


def void_summary_text(subpockets: List[Dict], mode: str) -> str:
    if not subpockets:
        return "No unoccupied sub-pockets detected."
    lines = [f"**Mode**: {mode}  |  **{len(subpockets)} sub-pocket(s) detected**\n"]
    for sp in subpockets:
        sc = sp["size_class"]
        lines.append(
            f"- **Sub-pocket {sp['sub_pocket_id']}** "
            f"({len(sp['residue_labels'])} residues, "
            f"r≈{sp['available_radius_A']:.1f} Å, "
            f"~{sp['void_volume_est_A3']:.0f} ų)  →  "
            f"**{sc}** fragments fit  ·  e.g. *{SIZE_EXAMPLES.get(sc, '')}*"
        )
    return "\n".join(lines)
