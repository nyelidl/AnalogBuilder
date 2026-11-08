"""Single-bond BRICS trimming and replacement; the append generator is unchanged.

Cuts come from RDKit BRICS. Generic [*] donors are accepted only when the
resulting junction is itself recognised by BRICS with the retained environment.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS, Crippen, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.DataStructs import TanimotoSimilarity

import core

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BRICSEdit:
    key: str
    parent_smiles: str
    atom_a: int
    atom_b: int
    bond_index: int
    keep_anchor: int
    keep_label: str
    remove_label: str
    retained: Chem.Mol
    removed: Chem.Mol
    retained_atoms: tuple
    removed_atoms: tuple


def _dummy(mol):
    atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(atoms) != 1 or atoms[0].GetDegree() != 1 or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("A replacement must have one connected component and one monovalent dummy atom.")
    return atoms[0]


def _complete(mol):
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        raise ValueError("The edited molecule is not a complete single component.")
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("The edit produced a radical.")
    for atom in mol.GetAtoms():
        for prop in ("_ab_keep", "_ab_donor"):
            if atom.HasProp(prop):
                atom.ClearProp(prop)
    return mol


def cap_with_hydrogen(fragment):
    """Remove a monovalent dummy by H capping, preserving surviving stereochemistry."""
    rw = Chem.RWMol(Chem.Mol(fragment))
    dummy = _dummy(rw)
    # Trimming changes the junction to an ordinary H, not an isotope or radical.
    rw.ReplaceAtom(dummy.GetIdx(), Chem.Atom(1))
    return _complete(Chem.RemoveHs(rw.GetMol()))


def find_brics_edits(parent):
    if parent is None or len(Chem.GetMolFrags(parent)) != 1:
        raise ValueError("BRICS edits require a single connected parent; select the molecular component explicitly.")
    if any(a.GetAtomicNum() == 0 or a.GetNumRadicalElectrons() for a in parent.GetAtoms()):
        raise ValueError("The parent must be a complete, closed-shell molecule without dummy atoms.")
    edits = []
    for (a, b), labels in BRICS.FindBRICSBonds(parent):
        bond = parent.GetBondBetweenAtoms(a, b)
        # First version handles single-junction edits only; no alkene or ring-bond surgery.
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        broken = BRICS.BreakBRICSBonds(parent, bonds=[((a, b), labels)])
        mapping = []
        pieces = Chem.GetMolFrags(broken, asMols=True, fragsMolAtomMapping=mapping)
        if len(pieces) != 2:
            continue
        for anchor, keep_label, remove_label in ((a, labels[0], labels[1]), (b, labels[1], labels[0])):
            keep_index = next(i for i, atoms in enumerate(mapping) if anchor in atoms)
            other = 1 - keep_index
            retained = Chem.Mol(pieces[keep_index])
            removed = Chem.Mol(pieces[other])
            _dummy(retained)
            _dummy(removed)
            edits.append(BRICSEdit(
                key=f"{min(a, b)}-{max(a, b)}:keep-{anchor}",
                parent_smiles=Chem.MolToSmiles(parent, canonical=False, isomericSmiles=True),
                atom_a=a, atom_b=b,
                bond_index=bond.GetIdx(), keep_anchor=anchor,
                keep_label=keep_label, remove_label=remove_label,
                retained=retained, removed=removed,
                retained_atoms=tuple(i for i in mapping[keep_index] if i < parent.GetNumAtoms()),
                removed_atoms=tuple(i for i in mapping[other] if i < parent.GetNumAtoms()),
            ))
    return sorted(edits, key=lambda e: (-e.retained.GetNumHeavyAtoms(), e.key))


def replace_brics_fragment(edit, replacement_smiles):
    """Join at the chosen cut and validate the actual product junction, not just donor labels."""
    donor = Chem.MolFromSmiles(replacement_smiles)
    if donor is None:
        raise ValueError("Invalid replacement SMILES.")
    kept = Chem.Mol(edit.retained)
    kd, dd = _dummy(kept), _dummy(donor)
    if kd.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE or dd.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE:
        raise ValueError("Only single-bond BRICS replacements are supported.")
    donor_label = dd.GetIsotope()
    # molzip manages atom-index changes and tetrahedral parity at the broken bond.
    tag = max(a.GetAtomMapNum() for m in (kept, donor) for a in m.GetAtoms()) + 1
    kd.SetAtomMapNum(tag)
    dd.SetAtomMapNum(tag)
    kd.GetNeighbors()[0].SetBoolProp("_ab_keep", True)
    dd.GetNeighbors()[0].SetBoolProp("_ab_donor", True)
    product = Chem.molzip(kept, donor)
    if product is None:
        raise ValueError("The cut and replacement could not be joined.")
    Chem.SanitizeMol(product)
    keep_idx = next(a.GetIdx() for a in product.GetAtoms() if a.HasProp("_ab_keep"))
    donor_idx = next(a.GetIdx() for a in product.GetAtoms() if a.HasProp("_ab_donor"))
    compatible = False
    for (a, b), (la, lb) in BRICS.FindBRICSBonds(product):
        if (a, b) == (keep_idx, donor_idx):
            kl, dl = la, lb
        elif (a, b) == (donor_idx, keep_idx):
            kl, dl = lb, la
        else:
            continue
        if kl == edit.keep_label and (not donor_label or str(donor_label) == dl):
            compatible = True
            break
    if not compatible:
        raise ValueError("The proposed replacement junction is not BRICS-compatible.")
    return _complete(product)


def draw_edit_svg(parent, edit):
    mol = Chem.Mol(parent)
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(720, 420)
    drawer.drawOptions().addAtomIndices = True
    colours = {i: (0.55, 0.85, 0.65) for i in edit.retained_atoms}
    colours.update({i: (1.0, 0.72, 0.40) for i in edit.removed_atoms})
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=list(colours), highlightAtomColors=colours,
        highlightBonds=[edit.bond_index], highlightBondColors={edit.bond_index: (0.95, 0.15, 0.15)},
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


RANK_OPTIONS = {
    "Similarity to parent": ("sim", False),
    "Drug-likeness (QED)": ("QED", False),
    "Solubility (ESOL)": ("ESOL", False),
    "Lower molecular weight": ("MW", True),
    "Synthetic-accessibility estimate": ("SA", True),
}


def generate_brics_analogs(parent, edits, fragments, mode="replace", max_analogs=100,
                           max_attempts=5000, max_MW=600.0, avoid_opts=None,
                           rank_by="Similarity to parent", on_progress=None):
    if mode not in {"trim", "replace"}:
        raise ValueError("BRICS mode must be trim or replace.")
    if rank_by not in RANK_OPTIONS:
        raise ValueError("Unknown BRICS ranking property.")
    if max_attempts < 1 or max_analogs < 1:
        raise ValueError("Generation limits must be positive.")
    # Validate edits against this exact parent; never apply stale atom-index selections.
    current = {e.key: e for e in find_brics_edits(parent)}
    ordered_parent = Chem.MolToSmiles(parent, canonical=False, isomericSmiles=True)
    edits = list(edits)
    if any(isinstance(e, BRICSEdit) and e.parent_smiles != ordered_parent for e in edits):
        raise ValueError("The BRICS cut belongs to a different parent or atom ordering.")
    keys = [e.key if isinstance(e, BRICSEdit) else e for e in edits]
    if not keys or any(key not in current for key in keys):
        raise ValueError("Select a BRICS cut for the current parent before generating.")
    avoid_opts = avoid_opts if avoid_opts is not None else {k: True for k in core.AVOID_SMARTS}
    alerts = [Chem.MolFromSmarts(s) for k, s in core.AVOID_SMARTS.items() if avoid_opts.get(k)]
    parent_smi = Chem.MolToSmiles(parent, isomericSmiles=True)
    parent_fp = AllChem.GetMorganFingerprintAsBitVect(parent, 2, nBits=2048)
    seen, rows = {parent_smi}, []
    counts = dict(attempted=0, incompatible_or_invalid=0, duplicate=0, MW=0, formal_charge=0, SMARTS=0)
    capped = False
    candidates = list(fragments) if mode == "replace" else [None]
    total = min(len(keys) * len(candidates), max_attempts)
    if on_progress:
        on_progress(0, total)
    for key in keys:
        edit = current[key]
        for fragment in candidates:
            if counts["attempted"] >= max_attempts:
                capped = True
                break
            counts["attempted"] += 1
            if on_progress and (counts["attempted"] % 50 == 0 or counts["attempted"] == total):
                on_progress(counts["attempted"], total)
            try:
                product = (cap_with_hydrogen(edit.retained) if mode == "trim"
                           else replace_brics_fragment(edit, fragment.smiles))
            except (ValueError, RuntimeError) as exc:
                counts["incompatible_or_invalid"] += 1
                _LOG.debug("BRICS candidate skipped for %s: %s", key, exc)
                continue
            smi = Chem.MolToSmiles(product, isomericSmiles=True)
            if smi in seen:
                counts["duplicate"] += 1
                continue
            if Descriptors.MolWt(product) > max_MW:
                counts["MW"] += 1
                continue
            if abs(Chem.GetFormalCharge(product)) > 1:
                counts["formal_charge"] += 1
                continue
            if any(q is not None and product.HasSubstructMatch(q) for q in alerts):
                counts["SMARTS"] += 1
                continue
            seen.add(smi)
            fp = AllChem.GetMorganFingerprintAsBitVect(product, 2, nBits=2048)
            # Report undefined atom and E/Z stereo rather than assigning arbitrary stereoisomers.
            undefined = [s for s in Chem.FindPotentialStereo(product)
                         if s.specified != Chem.StereoSpecified.Specified]
            unassigned = sum(str(s.type).startswith("Atom_") for s in undefined)
            unassigned_bonds = sum(str(s.type).startswith("Bond_") for s in undefined)
            rows.append(dict(
                smiles=smi, change=f"BRICS {mode}: {key}" + (f" + {fragment.name}" if fragment else " + H"),
                mode=f"brics_{mode}", sites=f"{edit.atom_a},{edit.atom_b}", n_sites=1,
                cut_bond=f"{edit.atom_a}-{edit.atom_b}", retained_anchor=edit.keep_anchor,
                retained_brics_type=edit.keep_label,
                removed_fragment=Chem.MolToSmiles(cap_with_hydrogen(edit.removed), isomericSmiles=True),
                fragment_name=fragment.name if fragment else "hydrogen cap",
                fragment_smiles=fragment.smiles if fragment else "[H]",
                fragment_category=fragment.category if fragment else "trim",
                fragment_size_class=core.infer_fragment_size_class(fragment) if fragment else "small",
                fragment_heavy_atoms=core._frag_heavy(fragment) if fragment else 0,
                MW=round(Descriptors.MolWt(product), 1), logP=round(Crippen.MolLogP(product), 2),
                TPSA=round(rdMolDescriptors.CalcTPSA(product), 1),
                HBD=rdMolDescriptors.CalcNumHBD(product), HBA=rdMolDescriptors.CalcNumHBA(product),
                QED=round(QED.qed(product), 3), ESOL=round(core.esol_logS(product), 2),
                SA=round(core.sa_score(product), 2), sim=round(TanimotoSimilarity(parent_fp, fp), 3),
                unspecified_stereocenters=unassigned, unspecified_stereobonds=unassigned_bonds,
            ))
        if capped:
            break
    if on_progress:
        on_progress(counts["attempted"], total)
    df = pd.DataFrame(rows)
    if not df.empty:
        column, ascending = RANK_OPTIONS[rank_by]
        df = df.sort_values([column, "smiles"], ascending=[ascending, True], kind="stable").head(max_analogs).reset_index(drop=True)
    df.attrs.update(filter_counts=counts, attempt_cap_reached=capped, rank_by=rank_by)
    return df
