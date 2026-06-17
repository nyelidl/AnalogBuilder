"""
core.py — Chemistry and computation backend for the Analog Designer application.

Contains:
  - Fragment library (Frag dataclass + LIBRARY)
  - Property calculators (SA score, ESOL, Morgan FP)
  - Analog generation (attach, attach_to_sites, generate_analogs)
  - Pocket analysis (distance-shell, fpocket alpha-spheres)
  - Docking helpers (PDB splitting, ACD command builder, score parsing)
  - PLIP / cIFP interaction fingerprints
  - 3D ligand file generation
"""

from __future__ import annotations

import glob
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    QED,
    rdMolDescriptors,
)
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.DataStructs import TanimotoSimilarity

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

_SA_OK = False
try:
    from rdkit.Chem import RDConfig
    import sys as _sys
    _sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # type: ignore
    _SA_OK = True
except Exception:
    pass


def sa_score(mol: Chem.Mol) -> float:
    """Synthetic-accessibility score 1 (easy) .. 10 (hard)."""
    if _SA_OK:
        try:
            return float(sascorer.calculateScore(mol))
        except Exception:
            pass
    nr = rdMolDescriptors.CalcNumRings(mol)
    nst = len(
        Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    )
    mw = Descriptors.MolWt(mol)
    sp3 = rdMolDescriptors.CalcFractionCSP3(mol)
    return max(1.0, min(10.0, 1.5 + 0.6 * nr + 0.7 * nst + mw / 250.0 + sp3))


def esol_logS(mol: Chem.Mol) -> float:
    """Delaney ESOL aqueous logS estimate."""
    clogp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ap = (
        len(mol.GetAromaticAtoms()) / mol.GetNumHeavyAtoms()
        if mol.GetNumHeavyAtoms()
        else 0.0
    )
    return 0.16 - 0.63 * clogp - 0.0062 * mw + 0.066 * rb - 0.74 * ap


def morgan(mol: Chem.Mol, r: int = 2, n: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=n)


# ---------------------------------------------------------------------------
# Fragment library
# ---------------------------------------------------------------------------

G = lambda **k: {
    "potency": 0,
    "selectivity": 0,
    "solubility": 0,
    "metabolic": 0,
    "synthesis": 0,
    "novelty": 0,
    **k,
}

CATEGORY_BASE_GOALS: Dict[str, Dict] = {
    "hydrophobic": G(potency=1, selectivity=1, solubility=-1, synthesis=1),
    "polar": G(potency=1, selectivity=1, solubility=1, synthesis=1),
    "basic": G(potency=1, selectivity=1, solubility=2, synthesis=0),
    "acidic": G(potency=1, selectivity=1, solubility=2, synthesis=0),
    "halogen": G(potency=1, selectivity=1, metabolic=2, solubility=-1, synthesis=1),
    "aromatic": G(potency=1, selectivity=2, solubility=-1, novelty=1),
    "solubility": G(solubility=2, selectivity=1, synthesis=0),
    "bioisostere": G(selectivity=1, metabolic=1, novelty=2, synthesis=-1),
}


def _merge_goals(category: str, **override) -> Dict:
    base = dict(CATEGORY_BASE_GOALS.get(category, G()))
    base.update(override)
    return base


@dataclass
class Frag:
    name: str
    smiles: str
    category: str
    goals: Dict
    tox: bool = False
    size_class: str = "auto"
    interaction_class: str = "auto"
    charge_class: str = "auto"
    source: str = "built_in"
    notes: str = ""

    @property
    def heavy(self) -> int:
        m = Chem.MolFromSmiles(self.smiles.replace("[*]", "[H]"))
        return m.GetNumHeavyAtoms() if m else 99


_FRAGMENT_ROWS = [
    ("methyl", "[*]C", "hydrophobic"),
    ("ethyl", "[*]CC", "hydrophobic"),
    ("n-propyl", "[*]CCC", "hydrophobic"),
    ("isopropyl", "[*]C(C)C", "hydrophobic"),
    ("n-butyl", "[*]CCCC", "hydrophobic"),
    ("tert-butyl", "[*]C(C)(C)C", "hydrophobic"),
    ("cyclopropyl", "[*]C1CC1", "hydrophobic"),
    ("cyclobutyl", "[*]C1CCC1", "hydrophobic"),
    ("cyclopentyl", "[*]C1CCCC1", "hydrophobic"),
    ("cyclohexyl", "[*]C1CCCCC1", "hydrophobic"),
    ("vinyl", "[*]C=C", "hydrophobic"),
    ("ethynyl", "[*]C#C", "hydrophobic"),
    ("hydroxyl", "[*]O", "polar"),
    ("methoxy", "[*]OC", "polar"),
    ("ethoxy", "[*]OCC", "polar"),
    ("isopropoxy", "[*]OC(C)C", "polar"),
    ("hydroxymethyl", "[*]CO", "polar"),
    ("2-hydroxyethyl", "[*]CCO", "polar"),
    ("acetyl", "[*]C(C)=O", "polar"),
    ("acetamido", "[*]NC(C)=O", "polar"),
    ("amide(C(=O)NH2)", "[*]C(N)=O", "polar"),
    ("N-methylamide", "[*]C(=O)NC", "polar"),
    ("urea", "[*]NC(=O)N", "polar"),
    ("methylsulfonyl", "[*]S(C)(=O)=O", "polar"),
    ("sulfonamide-NH2", "[*]S(N)(=O)=O", "polar"),
    ("amino", "[*]N", "basic"),
    ("methylamino", "[*]NC", "basic"),
    ("dimethylamino", "[*]N(C)C", "basic"),
    ("azetidine", "[*]N1CCC1", "basic"),
    ("pyrrolidine", "[*]N1CCCC1", "basic"),
    ("piperidine", "[*]N1CCCCC1", "basic"),
    ("morpholine", "[*]N1CCOCC1", "basic"),
    ("piperazine", "[*]N1CCNCC1", "basic"),
    ("N-methylpiperazine", "[*]N1CCN(C)CC1", "basic"),
    ("carboxyl", "[*]C(=O)O", "acidic"),
    ("carboxymethyl", "[*]CC(=O)O", "acidic"),
    ("sulfonic-acid", "[*]S(=O)(=O)O", "acidic"),
    ("sulfonamide", "[*]S(N)(=O)=O", "acidic"),
    ("tetrazole", "[*]c1nnn[nH]1", "acidic"),
    ("hydroxamic-acid", "[*]C(=O)NO", "acidic"),
    ("fluoro", "[*]F", "halogen"),
    ("chloro", "[*]Cl", "halogen"),
    ("bromo", "[*]Br", "halogen"),
    ("iodo", "[*]I", "halogen"),
    ("cyano", "[*]C#N", "halogen"),
    ("trifluoromethyl", "[*]C(F)(F)F", "halogen"),
    ("difluoromethyl", "[*]C(F)F", "halogen"),
    ("trifluoromethoxy", "[*]OC(F)(F)F", "halogen"),
    ("phenyl", "[*]c1ccccc1", "aromatic"),
    ("benzyl", "[*]Cc1ccccc1", "aromatic"),
    ("4-fluorophenyl", "[*]c1ccc(F)cc1", "aromatic"),
    ("4-chlorophenyl", "[*]c1ccc(Cl)cc1", "aromatic"),
    ("4-methylphenyl", "[*]c1ccc(C)cc1", "aromatic"),
    ("4-methoxyphenyl", "[*]c1ccc(OC)cc1", "aromatic"),
    ("pyridin-2-yl", "[*]c1ccccn1", "aromatic"),
    ("pyridin-3-yl", "[*]c1cccnc1", "aromatic"),
    ("pyridin-4-yl", "[*]c1ccncc1", "aromatic"),
    ("thiophen-2-yl", "[*]c1cccs1", "aromatic"),
    ("furan-2-yl", "[*]c1ccco1", "aromatic"),
    ("imidazol-1-yl", "[*]n1ccnc1", "aromatic"),
    ("pyrazol-1-yl", "[*]n1cccn1", "aromatic"),
    ("thiazol-2-yl", "[*]c1nccs1", "aromatic"),
    ("benzimidazolyl", "[*]c1nc2ccccc2[nH]1", "aromatic"),
    ("indol-3-yl", "[*]c1c[nH]c2ccccc12", "aromatic"),
    ("2-hydroxyethoxy", "[*]OCCO", "solubility"),
    ("PEG2", "[*]OCCOC", "solubility"),
    ("morpholinoethyl", "[*]CCN1CCOCC1", "solubility"),
    ("morpholine-carbonyl", "[*]C(=O)N1CCOCC1", "solubility"),
    ("N-methylpiperazine-carbonyl", "[*]C(=O)N1CCN(C)CC1", "solubility"),
    ("oxetan-3-yl", "[*]C1COC1", "bioisostere"),
    ("azetidin-3-yl", "[*]C1CNC1", "bioisostere"),
    ("tetrahydropyran-4-yl", "[*]C1CCOCC1", "bioisostere"),
    ("cyclopropyl-carbonyl", "[*]C(=O)C1CC1", "bioisostere"),
    ("difluorocyclopropyl", "[*]C1(F)CC1F", "bioisostere"),
    ("oxadiazole-methyl", "[*]Cc1nnco1", "bioisostere"),
    ("triazole-methyl", "[*]Cn1cncn1", "bioisostere"),
]

LIBRARY: List[Frag] = []
_seen_names: set = set()
for _name, _smi, _cat in _FRAGMENT_ROWS:
    if _name in _seen_names:
        raise ValueError(f"Duplicate fragment name: {_name}")
    _seen_names.add(_name)
    _mol = Chem.MolFromSmiles(_smi)
    if _mol is None:
        raise ValueError(f"Invalid fragment SMILES {_name}: {_smi}")
    LIBRARY.append(Frag(_name, _smi, _cat, _merge_goals(_cat)))

BUILTIN_LIBRARY: List[Frag] = list(LIBRARY)


# ---------------------------------------------------------------------------
# Fragment utilities
# ---------------------------------------------------------------------------

def infer_fragment_size_class(frag_or_smiles) -> str:
    if isinstance(frag_or_smiles, Frag):
        if frag_or_smiles.size_class and frag_or_smiles.size_class != "auto":
            return frag_or_smiles.size_class
        heavy = frag_or_smiles.heavy
    else:
        m = Chem.MolFromSmiles(str(frag_or_smiles).replace("[*]", "[H]"))
        heavy = m.GetNumHeavyAtoms() if m else 99
    if heavy <= 2:
        return "small"
    if heavy <= 5:
        return "medium"
    if heavy <= 10:
        return "large"
    return "extended"


def validate_fragment_smiles(smi: str) -> Tuple[bool, str]:
    if not isinstance(smi, str) or not smi.strip():
        return False, "empty"
    mol = Chem.MolFromSmiles(smi.strip())
    if mol is None:
        return False, "RDKit parse failed"
    ndummy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)
    if ndummy != 1:
        return False, f"needs exactly one [*], found {ndummy}"
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        return False, f"sanitize failed: {e}"
    return True, "ok"


def infer_charge_class(smi: str) -> str:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return "unknown"
    q = Chem.GetFormalCharge(mol)
    if q > 0:
        return "basic_or_cationic"
    if q < 0:
        return "acidic_or_anionic"
    if any(x in smi for x in ["C(=O)O", "S(=O)(=O)O", "n[nH]nn"]):
        return "acidic_possible"
    if any(x in smi for x in ["N", "n1", "n2"]):
        return "basic_or_hbonding_possible"
    return "neutral"


def annotate_library(lib: List[Frag]) -> None:
    for f in lib:
        if f.size_class == "auto":
            f.size_class = infer_fragment_size_class(f)
        if f.charge_class == "auto":
            f.charge_class = infer_charge_class(f.smiles)


annotate_library(BUILTIN_LIBRARY)

AVOID_SMARTS = {
    "nitro": "[N+](=O)[O-]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "reactive_acylhalide": "[CX3](=O)[F,Cl,Br,I]",
    "azide": "[N-]=[N+]=N",
    "michael_acceptor": "[CX3]=[CX3][CX3]=O",
    "epoxide": "C1OC1",
}


# ---------------------------------------------------------------------------
# Molecule drawing
# ---------------------------------------------------------------------------

def draw_mol_svg(mol: Chem.Mol, highlight: Optional[List[int]] = None, size=(560, 460)) -> str:
    highlight = list(highlight or [])
    d = rdMolDraw2D.MolDraw2DSVG(*size)
    o = d.drawOptions()
    o.addAtomIndices = True
    o.annotationFontScale = 0.8
    rdMolDraw2D.PrepareAndDrawMolecule(
        d,
        mol,
        highlightAtoms=highlight,
        highlightAtomColors={i: (1.0, 0.6, 0.6) for i in highlight},
    )
    d.FinishDrawing()
    return d.GetDrawingText()


def attachable_atom_indices(mol: Chem.Mol, carbon_only: bool = False) -> List[int]:
    return [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetTotalNumHs() > 0 and (not carbon_only or a.GetAtomicNum() == 6)
    ]


# ---------------------------------------------------------------------------
# Analog generation
# ---------------------------------------------------------------------------

def attach(parent: Chem.Mol, atom_idx: int, frag_smiles: str) -> Optional[Chem.Mol]:
    """Attach a [*]-fragment to parent atom by replacing one implicit H."""
    atom_idx = int(atom_idx)
    if parent.GetAtomWithIdx(atom_idx).GetTotalNumHs() == 0:
        return None
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None:
        return None
    dummies = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    dummy_idx = dummies[0]
    nbrs = frag.GetAtomWithIdx(dummy_idx).GetNeighbors()
    if len(nbrs) != 1:
        return None
    nbr_idx = nbrs[0].GetIdx()
    combo = Chem.CombineMols(parent, frag)
    rw = Chem.RWMol(combo)
    off = parent.GetNumAtoms()
    rw.AddBond(atom_idx, nbr_idx + off, Chem.BondType.SINGLE)
    rw.RemoveAtom(dummy_idx + off)
    m = rw.GetMol()
    try:
        Chem.SanitizeMol(m)
        AllChem.Compute2DCoords(m)
    except Exception:
        return None
    return m


def attach_to_sites(
    parent: Chem.Mol, atom_indices: List[int], frag_smiles: str
) -> Optional[Chem.Mol]:
    m = Chem.Mol(parent)
    for atom_idx in atom_indices:
        m = attach(m, int(atom_idx), frag_smiles)
        if m is None:
            return None
    return m


def _frag_heavy(f) -> int:
    if hasattr(f, "heavy"):
        try:
            return int(f.heavy)
        except Exception:
            pass
    smi = getattr(f, "smiles", "")
    mol = Chem.MolFromSmiles(smi)
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in [0, 1]) if mol else 999


def generate_analogs(
    parent: Chem.Mol,
    selected_atoms: List[int],
    chosen_frags: List[Frag],
    site_groups: Optional[List[Tuple]] = None,
    weights: Optional[Dict] = None,
    avoid_opts: Optional[Dict] = None,
    max_MW: float = 600.0,
    max_analogs: int = 100,
    rank_by: str = "Balanced (100-pt weights)",
    parent_name: str = "parent",
) -> pd.DataFrame:
    """
    Generate analogs by attaching fragments to selected sites.
    Returns a DataFrame with property columns.
    """
    if weights is None:
        weights = {k: 1 / 6 for k in ["potency", "selectivity", "solubility", "metabolic", "synthesis", "novelty"]}
    if avoid_opts is None:
        avoid_opts = {k: True for k in AVOID_SMARTS}

    avoid_q = {
        k: Chem.MolFromSmarts(v)
        for k, v in AVOID_SMARTS.items()
        if avoid_opts.get(k, False)
    }

    if site_groups is None:
        site_groups = [(s,) for s in selected_atoms]

    parent_can = Chem.MolToSmiles(parent)
    pfp = AllChem.GetMorganFingerprintAsBitVect(parent, 2, nBits=2048)

    rows, seen = [], {parent_can}
    filter_counts = {"attach_failed": 0, "duplicate": 0, "MW": 0, "formal_charge": 0, "SMARTS": 0, "SA": 0}

    def _passes(mol):
        if Descriptors.MolWt(mol) > max_MW:
            return False, "MW"
        if abs(Chem.GetFormalCharge(mol)) > 1:
            return False, "formal_charge"
        for q in avoid_q.values():
            if q and mol.HasSubstructMatch(q):
                return False, "SMARTS"
        return True, "passed"

    for sites in site_groups:
        for f in chosen_frags:
            m = attach_to_sites(parent, list(sites), f.smiles)
            if m is None:
                filter_counts["attach_failed"] += 1
                continue
            can = Chem.MolToSmiles(m)
            if can in seen:
                filter_counts["duplicate"] += 1
                continue
            ok, reason = _passes(m)
            if not ok:
                filter_counts[reason] = filter_counts.get(reason, 0) + 1
                continue
            seen.add(can)
            fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
            site_label = ",".join(map(str, sites))
            rows.append(dict(
                smiles=can,
                change=(
                    f"concerted@{site_label}+{f.name}"
                    if len(sites) > 1
                    else f"@{site_label}+{f.name}"
                ),
                mode=("concerted" if len(sites) > 1 else "individual"),
                sites=site_label,
                n_sites=len(sites),
                fragment_name=f.name,
                fragment_smiles=f.smiles,
                fragment_category=f.category,
                fragment_size_class=infer_fragment_size_class(f),
                fragment_heavy_atoms=_frag_heavy(f),
                MW=round(Descriptors.MolWt(m), 1),
                logP=round(Crippen.MolLogP(m), 2),
                TPSA=round(rdMolDescriptors.CalcTPSA(m), 1),
                HBD=rdMolDescriptors.CalcNumHBD(m),
                HBA=rdMolDescriptors.CalcNumHBA(m),
                QED=round(QED.qed(m), 3),
                ESOL=round(esol_logS(m), 2),
                SA=round(sa_score(m), 2),
                sim=round(TanimotoSimilarity(pfp, fp), 3),
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Ranking
    def _norm(s):
        s = pd.Series(s).astype(float)
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5

    balanced = (
        weights.get("solubility", 0) * _norm(df.ESOL)
        + weights.get("synthesis", 0) * (1 - _norm(df.SA))
        + weights.get("novelty", 0) * (1 - _norm(df.sim))
        + weights.get("metabolic", 0) * (1 - _norm((df.logP - 2.5).abs()))
        + (weights.get("potency", 0) + weights.get("selectivity", 0)) * _norm(df.QED)
    )
    df["balanced"] = balanced.round(3)
    df["binding_proxy"] = (_norm(df.logP.clip(upper=4)) + _norm(df.QED)).round(3)

    rank_col_map = {
        "Balanced (100-pt weights)": ("balanced", False),
        "Similarity to parent": ("sim", False),
        "Solubility (ESOL)": ("ESOL", False),
        "ADMET (QED)": ("QED", False),
        "Synthetic feasibility": ("SA", True),
        "Binding proxy (heuristic)": ("binding_proxy", False),
    }
    col, asc = rank_col_map.get(rank_by, ("balanced", False))
    df = df.sort_values(col, ascending=asc).head(max_analogs).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Pocket residue → fragment suggestion
# ---------------------------------------------------------------------------

AA_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
    "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}
AA3_TO_ONE = {v: k for k, v in AA_ONE_TO_THREE.items()}

AA_TOKEN_TO_ONE = {
    **{v: k for k, v in AA_ONE_TO_THREE.items()},
    **{k: k for k in AA_ONE_TO_THREE},
    **{v.upper(): k for k, v in AA_ONE_TO_THREE.items()},
}

AA_TAGS: Dict[str, List[str]] = {
    "D": ["acidic_negative", "hbond_acceptor"],
    "E": ["acidic_negative", "hbond_acceptor"],
    "K": ["basic_positive", "hbond_donor"],
    "R": ["basic_positive", "hbond_donor"],
    "H": ["basic_positive", "hbond_donor", "hbond_acceptor", "aromatic"],
    "S": ["polar_hbond", "hbond_donor", "hbond_acceptor"],
    "T": ["polar_hbond", "hbond_donor", "hbond_acceptor"],
    "N": ["polar_hbond", "hbond_donor", "hbond_acceptor"],
    "Q": ["polar_hbond", "hbond_donor", "hbond_acceptor"],
    "Y": ["polar_hbond", "hbond_donor", "aromatic", "hydrophobic"],
    "C": ["polar_hbond", "hbond_donor", "sulfur_polarizable"],
    "A": ["hydrophobic"], "V": ["hydrophobic"], "L": ["hydrophobic"],
    "I": ["hydrophobic"], "M": ["hydrophobic", "sulfur_polarizable"],
    "P": ["hydrophobic", "shape_constraint"],
    "F": ["hydrophobic", "aromatic"],
    "W": ["hydrophobic", "aromatic", "hbond_donor"],
    "G": ["small_flexible"],
}

TAG_TO_DESIGN: Dict[str, Dict] = {
    "acidic_negative": {
        "pocket_property": "acidic / negative residue",
        "ligand_strategy": "add basic/cationic or H-bond donor groups",
        "fragment_names": ["amino", "dimethylamino", "piperazine", "morpholine", "piperidine"],
    },
    "basic_positive": {
        "pocket_property": "basic / positive residue",
        "ligand_strategy": "add acidic/anionic or strong H-bond acceptor groups",
        "fragment_names": ["carboxyl", "sulfonamide", "tetrazole", "cyano", "methylsulfonyl"],
    },
    "polar_hbond": {
        "pocket_property": "polar H-bond residue",
        "ligand_strategy": "add H-bond donor/acceptor groups",
        "fragment_names": ["hydroxyl", "hydroxymethyl", "amide(C(=O)NH2)", "methoxy", "methylsulfonyl"],
    },
    "hbond_donor": {
        "pocket_property": "residue can donate H-bond",
        "ligand_strategy": "add ligand H-bond acceptor groups",
        "fragment_names": ["methoxy", "cyano", "methylsulfonyl", "pyridin-3-yl"],
    },
    "hbond_acceptor": {
        "pocket_property": "residue can accept H-bond",
        "ligand_strategy": "add ligand H-bond donor groups",
        "fragment_names": ["hydroxyl", "amino", "amide(C(=O)NH2)", "sulfonamide"],
    },
    "hydrophobic": {
        "pocket_property": "hydrophobic residue",
        "ligand_strategy": "add small hydrophobic, aromatic, or halogen groups",
        "fragment_names": ["methyl", "ethyl", "isopropyl", "cyclopropyl", "chloro", "trifluoromethyl", "phenyl"],
    },
    "aromatic": {
        "pocket_property": "aromatic residue",
        "ligand_strategy": "add π-stacking or hydrophobic groups",
        "fragment_names": ["phenyl", "pyridin-3-yl", "thiophen-2-yl", "pyrazol-1-yl", "chloro"],
    },
    "sulfur_polarizable": {
        "pocket_property": "sulfur / polarizable residue",
        "ligand_strategy": "add soft hydrophobic/halogen groups",
        "fragment_names": ["chloro", "trifluoromethyl", "thiophen-2-yl", "phenyl", "methylsulfonyl"],
    },
    "shape_constraint": {
        "pocket_property": "shape-constraining residue",
        "ligand_strategy": "add compact conformationally restricted groups",
        "fragment_names": ["cyclopropyl", "oxetan-3-yl", "methyl"],
    },
    "small_flexible": {
        "pocket_property": "small/flexible residue",
        "ligand_strategy": "space may tolerate small growth",
        "fragment_names": ["methyl", "fluoro", "hydroxyl", "cyano"],
    },
}


def parse_pocket_residues(text: str) -> List[str]:
    found = []
    for raw in re.findall(r"[A-Za-z]{1,14}\d*", text or ""):
        letters = re.sub(r"\d+", "", raw).upper()
        if letters in AA_TOKEN_TO_ONE:
            found.append(AA_TOKEN_TO_ONE[letters])
    return found


def suggest_fragments_from_residues(
    residue_codes: List[str],
    active_library: List[Frag],
    max_suggestions: int = 6,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Frag]]:
    """
    Given residue one-letter codes, return:
      - strategy_df: why each fragment type is suggested
      - ratio_df: pocket property ratios
      - pocket_frags: Frag objects to use
    """
    tag_counts: Dict[str, int] = {}
    for aa in residue_codes:
        for tag in AA_TAGS.get(aa, []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    total_tags = sum(tag_counts.values()) or 1
    ratios = {k: round(100 * v / total_tags, 1) for k, v in sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)}

    ratio_df = pd.DataFrame([{"pocket_property_tag": k, "ratio_%": v} for k, v in ratios.items()])

    frag_score: Dict[str, float] = {}
    strategy_rows = []
    for tag, ratio in ratios.items():
        info = TAG_TO_DESIGN.get(tag)
        if not info:
            continue
        strategy_rows.append({
            "property_tag": tag,
            "ratio_%": ratio,
            "pocket_property": info["pocket_property"],
            "ligand_strategy": info["ligand_strategy"],
            "suggested_fragment_examples": ", ".join(info["fragment_names"][:5]),
        })
        for fname in info["fragment_names"]:
            frag_score[fname] = frag_score.get(fname, 0.0) + ratio

    strategy_df = pd.DataFrame(strategy_rows).sort_values("ratio_%", ascending=False) if strategy_rows else pd.DataFrame()

    lib_by_name = {f.name: f for f in active_library}
    ordered = sorted(frag_score.items(), key=lambda kv: kv[1], reverse=True)
    pocket_frags: List[Frag] = []
    seen: set = set()
    for name, _ in ordered:
        if name in lib_by_name and name not in seen:
            pocket_frags.append(lib_by_name[name])
            seen.add(name)
    pocket_frags = pocket_frags[:max_suggestions]
    return strategy_df, ratio_df, pocket_frags


# ---------------------------------------------------------------------------
# PDB / structure helpers
# ---------------------------------------------------------------------------

AA3_STRUCT = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL SEC PYL ASX GLX HID HIE HIP".split()
)
EXCLUDE_HET = set(
    "HOH WAT DOD NA CL K MG MN ZN CA FE CU CO NI CD HG SO4 PO4 HPO4 ACT ACE EDO GOL PEG DMS DMSO MPD TRS BME MSE".split()
)


def _safe_file_token(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x).strip() or "file")


def _pdb_xyz(line: str) -> np.ndarray:
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)


def _pdb_is_h(line: str) -> bool:
    elem = line[76:78].strip().upper()
    name = line[12:16].strip().upper()
    return elem == "H" or name.startswith("H")


def _pdb_reskey(line: str) -> Tuple:
    return (line[21].strip() or "_", line[17:20].strip().upper(), line[22:26].strip(), line[26].strip())


def download_pdb(pdb_id: str, out_dir: Path) -> str:
    pdb_id = pdb_id.strip().upper()
    assert re.fullmatch(r"[A-Za-z0-9]{4}", pdb_id), "PDB ID must be 4 characters."
    out = out_dir / f"{pdb_id}.pdb"
    if not out.exists() or out.stat().st_size < 1000:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(url, out)
    return str(out)


def cif_to_pdb_if_needed(path: str) -> str:
    path = Path(path)
    if path.suffix.lower() not in [".cif", ".mmcif"]:
        return str(path)
    out = path.with_suffix(".pdb")
    try:
        import gemmi  # type: ignore
        st = gemmi.read_structure(str(path))
        st.write_pdb(str(out))
        return str(out)
    except Exception as e:
        raise RuntimeError("Could not convert CIF to PDB. Install gemmi or provide PDB.") from e


def detect_ligand_candidates(pdb_path: str) -> Tuple[pd.DataFrame, Dict]:
    groups: Dict = {}
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("HETATM"):
                continue
            resn = line[17:20].strip().upper()
            if resn in EXCLUDE_HET or resn in AA3_STRUCT:
                continue
            key = _pdb_reskey(line)
            groups.setdefault(key, {"atoms": 0, "heavy": 0})
            groups[key]["atoms"] += 1
            groups[key]["heavy"] += 0 if _pdb_is_h(line) else 1
    rows = []
    for (chain, resn, resi, icode), val in groups.items():
        rows.append({"chain": chain, "resname": resn, "resnum": resi, "icode": icode,
                     "atoms": val["atoms"], "heavy_atoms": val["heavy"]})
    df = pd.DataFrame(rows).sort_values(["heavy_atoms", "atoms"], ascending=False) if rows else pd.DataFrame()
    return df, groups


def split_protein_ligand(
    pdb_path: str,
    ligand_resname: str = "",
    work_dir: Optional[Path] = None,
) -> Tuple[str, Optional[str], pd.DataFrame]:
    """Split complex into protein-only PDB and reference ligand PDB."""
    work_dir = work_dir or Path(".")
    work_dir.mkdir(parents=True, exist_ok=True)
    ligand_resname = ligand_resname.strip().upper()
    candidates, groups = detect_ligand_candidates(pdb_path)
    chosen_key = None
    if ligand_resname:
        for key in groups:
            if key[1] == ligand_resname:
                chosen_key = key
                break
        if chosen_key is None:
            raise ValueError(f"Residue {ligand_resname} not found in HETATM candidates.")
    elif groups:
        chosen_key = max(groups, key=lambda k: groups[k]["heavy"])

    protein_lines, ligand_lines = [], []
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("ATOM"):
                protein_lines.append(line)
            elif line.startswith("HETATM") and chosen_key and _pdb_reskey(line) == chosen_key:
                ligand_lines.append(line)

    protein_out = str(work_dir / "protein_only.pdb")
    with open(protein_out, "w") as f:
        f.writelines(protein_lines)
        f.write("END\n")

    ligand_out = None
    if ligand_lines:
        ligand_out = str(work_dir / "reference_ligand.pdb")
        with open(ligand_out, "w") as f:
            f.writelines(ligand_lines)
            f.write("END\n")

    return protein_out, ligand_out, candidates


def combine_protein_ligand_pdb(protein_pdb: str, ligand_pdb: str, out_pdb: str) -> str:
    protein_lines = []
    with open(protein_pdb, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM"):
                protein_lines.append(line)
    ligand_lines = []
    with open(ligand_pdb, "r", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                newline = "HETATM" + line[6:17] + "LIG A 900" + line[26:]
                ligand_lines.append(newline)
    with open(out_pdb, "w") as f:
        f.writelines(protein_lines)
        f.writelines(ligand_lines)
        f.write("END\n")
    return out_pdb


def sdf_first_mol_to_pdb(sdf_path: str, out_pdb: str) -> str:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = next((m for m in suppl if m is not None), None)
    if mol is None:
        raise ValueError(f"No readable molecule in SDF: {sdf_path}")
    Chem.MolToPDBFile(mol, str(out_pdb))
    return str(out_pdb)


# ---------------------------------------------------------------------------
# Pocket distance-shell analysis
# ---------------------------------------------------------------------------

def read_complex_atoms_for_pocket(pdb_path: str) -> Tuple[List[Dict], List[Dict]]:
    protein, ligand = [], []
    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if _pdb_is_h(line):
                continue
            try:
                xyz = _pdb_xyz(line)
            except Exception:
                continue
            rec = {
                "record": line[:6].strip(),
                "atom_name": line[12:16].strip(),
                "resname": line[17:20].strip().upper(),
                "chain": line[21].strip() or "_",
                "resnum": line[22:26].strip(),
                "icode": line[26].strip(),
                "xyz": xyz,
            }
            if line.startswith("ATOM"):
                protein.append(rec)
            else:
                ligand.append(rec)
    return protein, ligand


def analyze_complex_distance_shell(
    complex_pdb: str,
    pocket_cutoff: float = 6.0,
    contact_cutoff: float = 4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
    protein, ligand = read_complex_atoms_for_pocket(complex_pdb)
    if not protein or not ligand:
        raise ValueError("Complex must contain protein ATOM records and ligand HETATM records.")
    lig_xyz = np.array([a["xyz"] for a in ligand], dtype=float)
    by_res: Dict = {}
    for a in protein:
        dmin = float(np.linalg.norm(lig_xyz - a["xyz"], axis=1).min())
        key = (a["chain"], a["resname"], a["resnum"], a["icode"])
        if key not in by_res or dmin < by_res[key]["min_dist_to_ligand_A"]:
            by_res[key] = {
                "key": key, "resname": a["resname"], "chain": a["chain"],
                "resnum": a["resnum"], "icode": a["icode"], "min_dist_to_ligand_A": dmin,
            }
    prot_rows = []
    for r in by_res.values():
        r["is_pocket_residue"] = r["min_dist_to_ligand_A"] <= pocket_cutoff
        r["is_contacted"] = r["min_dist_to_ligand_A"] <= contact_cutoff
        r["is_noncontact_growth_residue"] = r["is_pocket_residue"] and not r["is_contacted"]
        one = AA3_TO_ONE.get(r["resname"], "")
        r["aa_one"] = one
        r["property_tags"] = ",".join(AA_TAGS.get(one, [])) if one else ""
        r["residue_label"] = f"{r['resname']}{r['resnum']}:{r['chain']}"
        prot_rows.append(r)
    df = pd.DataFrame(prot_rows).sort_values("min_dist_to_ligand_A")
    return (
        df[df["is_pocket_residue"]].copy(),
        df[df["is_contacted"]].copy(),
        df[df["is_noncontact_growth_residue"]].copy(),
        ligand,
    )


# ---------------------------------------------------------------------------
# ACD docking command builder
# ---------------------------------------------------------------------------

def build_acd_dock_cmd(
    receptor: str,
    smiles: str,
    center: str = "auto",
    name: str = "ligand",
    ph: float = 7.4,
    output_dir: str = "docking_out",
    cx: float = 0.0,
    cy: float = 0.0,
    cz: float = 0.0,
    use_pkanet: bool = False,
    neutral: bool = False,
    save_poses: bool = True,
    extra_args: str = "",
) -> List[str]:
    cmd = ["acd", "dock", "--receptor", receptor, "--smiles", smiles,
           "--center", center, "--name", _safe_file_token(name),
           "--ph", str(ph), "-o", output_dir]
    if center == "manual":
        cmd.extend(["--cx", str(cx), "--cy", str(cy), "--cz", str(cz)])
    if use_pkanet:
        cmd.append("--pkanet")
    if neutral:
        cmd.append("--neutral")
    if save_poses:
        cmd.append("--save-poses")
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args.strip()))
    return cmd


def build_acd_batch_cmd(
    receptor: str,
    ligands_smi: str,
    output_dir: str = "docking_out",
    center: str = "auto",
    exhaustiveness: int = 8,
    num_poses: int = 10,
    ph: float = 7.4,
    box_x: float = 16.0,
    box_y: float = 16.0,
    box_z: float = 16.0,
    use_pkanet: bool = False,
    neutral: bool = False,
    extra_args: str = "",
) -> List[str]:
    cmd = ["acd", "batch", "--receptor", receptor, "--ligands", ligands_smi,
           "--output", output_dir, "--center", center,
           "-e", str(exhaustiveness), "-n", str(num_poses), "--ph", str(ph)]
    if use_pkanet:
        cmd.append("--pkanet")
    if neutral:
        cmd.append("--neutral")
    if extra_args.strip():
        cmd.extend(shlex.split(extra_args.strip()))
    return cmd


def run_command(cmd: List[str], log_path: Optional[str] = None) -> Tuple[int, str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    output = (proc.stdout or "") + (proc.stderr or "")
    if log_path:
        Path(log_path).write_text(output)
    return proc.returncode, output


def parse_acd_score_csvs(out_dir: str) -> Optional[Dict]:
    best_rows = []
    for csv_path in glob.glob(str(Path(out_dir) / "**" / "*.csv"), recursive=True):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if df.empty:
            continue
        score_cols = [c for c in df.columns if re.search(r"score|energy|affinity|binding", c, re.I)]
        if not score_cols:
            continue
        sc = score_cols[0]
        df[sc] = pd.to_numeric(df[sc], errors="coerce")
        df = df.dropna(subset=[sc])
        if df.empty:
            continue
        idx = df[sc].idxmin()
        row = df.loc[idx].to_dict()
        row["_score_csv"] = csv_path
        row["_score_col"] = sc
        best_rows.append(row)
    if not best_rows:
        return None
    best_rows.sort(key=lambda r: float(r.get(r.get("_score_col", ""), 9999)))
    return best_rows[0]


def find_pose_sdf(out_dir: str) -> Optional[str]:
    sdfs = sorted(glob.glob(str(Path(out_dir) / "**" / "*.sdf"), recursive=True))
    ranked = [p for p in sdfs if re.search(r"out|pose|dock|result", Path(p).name, re.I)] + sdfs
    return ranked[0] if ranked else None


# ---------------------------------------------------------------------------
# PLIP / distance-contact cIFP
# ---------------------------------------------------------------------------

def _parse_pdb_heavy_atoms(pdb_path: str) -> Tuple[List[Dict], List[Dict]]:
    protein, ligand = [], []
    with open(pdb_path, "r", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            elem = line[76:78].strip().upper() or line[12:16].strip()[0:1].upper()
            if elem == "H":
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except Exception:
                continue
            rec = {
                "record": line[:6].strip(),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "chain": line[21].strip() or "_",
                "resnum": line[22:26].strip(),
                "x": x, "y": y, "z": z,
            }
            if line.startswith("ATOM"):
                protein.append(rec)
            else:
                ligand.append(rec)
    return protein, ligand


def distance_contact_cifp(complex_pdb: str, cutoff: float = 4.0) -> List[str]:
    protein, ligand = _parse_pdb_heavy_atoms(complex_pdb)
    feats: set = set()
    c2 = float(cutoff) ** 2
    for pa in protein:
        for la in ligand:
            dx = pa["x"] - la["x"]
            dy = pa["y"] - la["y"]
            dz = pa["z"] - la["z"]
            if dx * dx + dy * dy + dz * dz <= c2:
                feats.add(f"CONTACT:{pa['chain']}:{pa['resname']}:{pa['resnum']}")
                break
    return sorted(feats)


def run_plip(complex_pdb: str, out_dir: str, name: str) -> Tuple[Optional[str], Optional[str]]:
    plip_exe = shutil.which("plipcmd") or shutil.which("plip")
    if plip_exe is None:
        return None, "PLIP executable not found"
    od = Path(out_dir) / name
    od.mkdir(parents=True, exist_ok=True)
    cmd = [plip_exe, "-f", str(complex_pdb), "-x", "-o", str(od)]
    res = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (od / "plip_log.txt").write_text(res.stdout)
    xmls = list(od.glob("*.xml")) + list(od.glob("**/*.xml"))
    if res.returncode != 0 or not xmls:
        return None, f"PLIP failed. Return={res.returncode}."
    return str(xmls[0]), None


def parse_plip_xml(xml_path: str) -> List[str]:
    feats: set = set()
    if not xml_path or not os.path.exists(xml_path):
        return []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception:
        return []
    type_map = {
        "hydrophobic_interaction": "HYDROPHOBIC",
        "hydrogen_bond": "HBOND",
        "water_bridge": "WATERBRIDGE",
        "salt_bridge": "SALTBRIDGE",
        "pi_stack": "PISTACK",
        "pi_cation_interaction": "PICATION",
        "halogen_bond": "HALOGEN",
        "metal_complex": "METAL",
    }
    for elem in root.iter():
        tag = elem.tag.split("}")[-1].lower()
        itype = next((v for k, v in type_map.items() if k in tag), None)
        if itype is None:
            continue
        vals = {c.tag.split("}")[-1].lower(): (c.text or "").strip() for c in elem.iter() if c.text}
        resnr = vals.get("resnr") or vals.get("resnum") or "NA"
        restype = vals.get("restype") or vals.get("resname") or "RES"
        reschain = vals.get("reschain") or vals.get("chain") or "_"
        feats.add(f"{itype}:{reschain}:{restype}:{resnr}")
    return sorted(feats)


# ---------------------------------------------------------------------------
# 3D ligand file generation
# ---------------------------------------------------------------------------

def build_3d_mol(smiles: str, seed: int = 42, mmff: bool = True) -> Tuple[Optional[Chem.Mol], str]:
    mol0 = Chem.MolFromSmiles(smiles)
    if mol0 is None:
        return None, "invalid_smiles"
    mol = Chem.AddHs(mol0)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            return None, "embed_failed"
    if mmff:
        try:
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            if props:
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=1000)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        except Exception:
            pass
    return mol, "ok"


def generate_3d_ligand_files(
    ligand_table: pd.DataFrame,
    out_dir: Path,
    formats: List[str] = ["SDF"],
    mmff: bool = True,
) -> pd.DataFrame:
    """
    ligand_table must have 'compound' and 'smiles' columns.
    Returns manifest DataFrame.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    obabel = shutil.which("obabel")
    combined_sdf = out_dir / "all_ligands_3d.sdf"
    writer = Chem.SDWriter(str(combined_sdf))
    rows = []
    for i, r in ligand_table.iterrows():
        compound = _safe_file_token(r.get("compound", f"ligand_{i+1}"))
        smiles = str(r.get("smiles", "")).strip()
        mol, status = build_3d_mol(smiles, seed=42 + int(i), mmff=mmff)
        sdf_p = pdb_p = mol2_p = None
        if mol:
            mol.SetProp("_Name", compound)
            if "SDF" in formats:
                sdf_p = str(out_dir / f"{compound}.sdf")
                w = Chem.SDWriter(sdf_p)
                w.write(mol)
                w.close()
            if "PDB" in formats:
                pdb_p = str(out_dir / f"{compound}.pdb")
                Chem.MolToPDBFile(mol, pdb_p)
            if "MOL2" in formats and obabel and sdf_p:
                mol2_p = str(out_dir / f"{compound}.mol2")
                subprocess.run([obabel, sdf_p, "-O", mol2_p], capture_output=True, check=False)
                if not os.path.exists(mol2_p):
                    mol2_p = None
            writer.write(mol)
        rows.append({"compound": compound, "smiles": smiles, "status": status,
                     "sdf": sdf_p, "pdb": pdb_p, "mol2": mol2_p})
    writer.close()
    return pd.DataFrame(rows)
