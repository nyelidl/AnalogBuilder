"""
external_fragment_library.py — Fetch fragment libraries from ChEMBL and ZINC
─────────────────────────────────────────────────────────────────────────────
Two sources:
  1. ChEMBL REST API   — rule-of-three fragments, paginated, real-time
  2. ZINC Fragments    — pre-curated fragment subset, cached locally

Both add [*] attachment points compatible with Analog Designer core.py Frag objects.
"""

from __future__ import annotations
import hashlib, json, os, re, time
from pathlib import Path
from typing import List, Optional, Tuple
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(os.environ.get("ANALOGBUILDER_CACHE", Path.home() / ".analogbuilder" / "fragment_cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{hashlib.md5(key.encode()).hexdigest()}.json"

def _load_cache(key: str, max_age_hours: int = 24) -> Optional[list]:
    p = _cache_path(key)
    if not p.exists():
        return None
    age = (time.time() - p.stat().st_mtime) / 3600
    if age > max_age_hours:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _save_cache(key: str, data: list) -> None:
    try:
        _cache_path(key).write_text(json.dumps(data))
    except Exception:
        pass

# ── SMILES validation + [*] attachment helper ─────────────────────────────────
def _add_attachment(smiles: str) -> Optional[str]:
    """
    Return attachment SMILES with [*] at the best attachment atom.
    Strategy: pick atom with highest degree that is not in a ring 
    (terminal / exo) — most chemically sensible.
    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Find best non-ring atom (highest valence available)
    best_idx = None
    best_score = -1
    for atom in mol.GetAtoms():
        if atom.IsInRing():
            continue
        score = atom.GetDegree()
        # Prefer carbon attachment
        bonus = 1 if atom.GetAtomicNum() == 6 else 0
        if score + bonus > best_score:
            best_score = score + bonus
            best_idx = atom.GetIdx()
    if best_idx is None:
        # Fall back to any atom
        best_idx = 0

    # Build edit mol, replace atom with [*] dummy
    edit = Chem.RWMol(mol)
    dummy = Chem.Atom(0)   # atomic num 0 = dummy [*]
    edit.ReplaceAtom(best_idx, dummy)
    try:
        smi = Chem.MolToSmiles(edit.GetMol())
        # Validate round-trip
        if Chem.MolFromSmiles(smi) is not None:
            return smi
    except Exception:
        pass
    return None

def _meets_ro3(smiles: str, max_mw=300, max_hbd=3, max_hba=3,
               max_logp=3, max_rtb=3) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return (
        Descriptors.ExactMolWt(mol) <= max_mw and
        rdMolDescriptors.CalcNumHBD(mol) <= max_hbd and
        rdMolDescriptors.CalcNumHBA(mol) <= max_hba and
        Descriptors.MolLogP(mol) <= max_logp and
        rdMolDescriptors.CalcNumRotatableBonds(mol) <= max_rtb
    )

def _categorise(smiles: str) -> str:
    """Assign fragment to one of Analog Designer's 8 categories."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "polar"
    has_aromatic  = any(a.GetIsAromatic() for a in mol.GetAtoms())
    has_halogen   = any(a.GetAtomicNum() in (9,17,35,53) for a in mol.GetAtoms())
    has_nitrogen  = any(a.GetAtomicNum() == 7 for a in mol.GetAtoms())
    has_oxygen    = any(a.GetAtomicNum() == 8 for a in mol.GetAtoms())
    has_acid      = mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OH]")) or \
                    mol.HasSubstructMatch(Chem.MolFromSmarts("[S](=O)(=O)[NH]"))
    has_basic_N   = mol.HasSubstructMatch(Chem.MolFromSmarts("[NX3;H0,H1;!$(NC=O)]"))
    mw            = Descriptors.ExactMolWt(mol)
    logp          = Descriptors.MolLogP(mol)
    hbd           = rdMolDescriptors.CalcNumHBD(mol)

    if has_acid:             return "acidic"
    if has_halogen:          return "halogen"
    if has_basic_N:          return "basic"
    if has_aromatic:         return "aromatic"
    if hbd >= 1 or has_oxygen: return "polar"
    if logp > 1.5:           return "hydrophobic"
    return "solubility"

# ── 1. ChEMBL API ─────────────────────────────────────────────────────────────
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
CHEMBL_PARAMS = {
    "molecule_properties__mw_freebase__lte": 300,
    "molecule_properties__rtb__lte": 3,
    "molecule_properties__hbd__lte": 3,
    "molecule_properties__hba__lte": 3,
    "molecule_properties__alogp__lte": 3,
    "molecule_type": "Small Molecule",
    "limit": 100,
    "offset": 0,
}

def fetch_chembl_fragments(
    max_results: int = 500,
    timeout: int = 15,
    progress_cb=None,
) -> List[dict]:
    """
    Fetch rule-of-three fragments from ChEMBL REST API.
    Returns list of dicts: {name, smiles, category, source, chembl_id, mw, logp}
    Uses local cache (24h) to avoid repeated API calls.
    """
    cache_key = f"chembl_ro3_{max_results}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    results = []
    params  = dict(CHEMBL_PARAMS)
    session = requests.Session()
    session.headers.update({"User-Agent": "AnalogDesigner/1.0 (research)"})
    fetched = 0

    while fetched < max_results:
        params["offset"] = fetched
        params["limit"]  = min(100, max_results - fetched)
        try:
            resp = session.get(CHEMBL_BASE, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            break

        mols = data.get("molecules", [])
        if not mols:
            break

        for m in mols:
            smi_raw = (m.get("molecule_structures") or {}).get("canonical_smiles", "")
            if not smi_raw:
                continue
            props   = m.get("molecule_properties") or {}
            chembl_id = m.get("molecule_chembl_id", "")
            # Validate + add attachment point
            att_smi = _add_attachment(smi_raw)
            if att_smi is None:
                continue
            results.append({
                "name":      chembl_id,
                "smiles":    att_smi,
                "smiles_orig": smi_raw,
                "category":  _categorise(smi_raw),
                "source":    "ChEMBL",
                "chembl_id": chembl_id,
                "mw":        float(props.get("mw_freebase", 0) or 0),
                "logp":      float(props.get("alogp", 0) or 0),
            })

        fetched += len(mols)
        if progress_cb:
            progress_cb(fetched, data.get("page_meta", {}).get("total_count", max_results))

        # Respect ChEMBL rate limit
        time.sleep(0.3)

        if len(mols) < params["limit"]:
            break

    _save_cache(cache_key, results)
    return results

# ── 2. ZINC Fragments ─────────────────────────────────────────────────────────
ZINC_FRAG_URL = "https://zinc.docking.org/substances/subsets/fragment.json"
ZINC_FRAG_SMI = "https://zinc.docking.org/substances/subsets/fragment.smi"

def fetch_zinc_fragments(
    max_results: int = 1000,
    timeout: int = 20,
    progress_cb=None,
) -> List[dict]:
    """
    Fetch ZINC fragment subset.
    Uses bulk .smi download when max_results > 200, JSON otherwise.
    Cache valid for 72h (ZINC fragments don't change often).
    """
    cache_key = f"zinc_frag_{max_results}"
    cached = _load_cache(cache_key, max_age_hours=72)
    if cached is not None:
        return cached

    results = []
    session = requests.Session()
    session.headers.update({"User-Agent": "AnalogDesigner/1.0 (research)"})

    if max_results <= 200:
        # JSON paginated
        fetched = 0
        offset  = 0
        while fetched < max_results:
            count = min(100, max_results - fetched)
            try:
                resp = session.get(
                    ZINC_FRAG_URL,
                    params={"count": count, "offset": offset},
                    timeout=timeout,
                )
                resp.raise_for_status()
                items = resp.json()
            except Exception:
                break

            if not items:
                break

            for item in items:
                smi_raw = item.get("smiles", "")
                zinc_id = item.get("zinc_id", f"ZINC_{fetched}")
                if not smi_raw or not _meets_ro3(smi_raw):
                    continue
                att_smi = _add_attachment(smi_raw)
                if att_smi is None:
                    continue
                results.append({
                    "name":      zinc_id,
                    "smiles":    att_smi,
                    "smiles_orig": smi_raw,
                    "category":  _categorise(smi_raw),
                    "source":    "ZINC",
                    "zinc_id":   zinc_id,
                    "mw":        float(item.get("mwt", 0) or 0),
                    "logp":      float(item.get("logp", 0) or 0),
                })
            fetched += len(items)
            offset  += len(items)
            if progress_cb:
                progress_cb(fetched, max_results)
            time.sleep(0.2)
    else:
        # Bulk SMI download
        try:
            resp = session.get(ZINC_FRAG_SMI, timeout=60, stream=True)
            resp.raise_for_status()
            lines_read = 0
            for line in resp.iter_lines():
                if lines_read >= max_results:
                    break
                line = line.decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                smi_raw = parts[0]
                zinc_id = parts[1] if len(parts) > 1 else f"ZINC_{lines_read}"
                if not _meets_ro3(smi_raw):
                    continue
                att_smi = _add_attachment(smi_raw)
                if att_smi is None:
                    continue
                results.append({
                    "name":      zinc_id,
                    "smiles":    att_smi,
                    "smiles_orig": smi_raw,
                    "category":  _categorise(smi_raw),
                    "source":    "ZINC",
                    "zinc_id":   zinc_id,
                    "mw":        0.0,
                    "logp":      0.0,
                })
                lines_read += 1
                if progress_cb and lines_read % 100 == 0:
                    progress_cb(lines_read, max_results)
        except Exception as e:
            pass

    _save_cache(cache_key, results)
    return results

# ── 3. Combined fetch ─────────────────────────────────────────────────────────
def fetch_external_fragments(
    sources: list = ("chembl", "zinc"),
    max_per_source: int = 500,
    timeout: int = 15,
    progress_cb=None,
) -> List[dict]:
    """
    Fetch from one or both sources, deduplicate by canonical SMILES.
    Returns merged list sorted by MW.
    """
    all_results = []
    seen_smi = set()

    for src in sources:
        if src == "chembl":
            frags = fetch_chembl_fragments(max_per_source, timeout, progress_cb)
        elif src == "zinc":
            frags = fetch_zinc_fragments(max_per_source, timeout, progress_cb)
        else:
            continue

        for f in frags:
            canon = Chem.MolToSmiles(Chem.MolFromSmiles(f["smiles_orig"])) if f.get("smiles_orig") else f["smiles"]
            if canon and canon not in seen_smi:
                seen_smi.add(canon)
                all_results.append(f)

    return sorted(all_results, key=lambda x: x.get("mw", 0))

# ── 4. Convert to core.Frag objects ──────────────────────────────────────────
def to_frag_objects(external_list: List[dict], core_module) -> list:
    """Convert fetch results to core.Frag objects for use in generate_analogs()."""
    frags = []
    for item in external_list:
        ok, _ = core_module.validate_fragment_smiles(item["smiles"])
        if ok:
            frags.append(core_module.Frag(
                name     = item["name"],
                smiles   = item["smiles"],
                category = item["category"],
                goals    = core_module.G(),
            ))
    return frags

# ── 5. Clear cache ────────────────────────────────────────────────────────────
def clear_cache():
    for f in CACHE_DIR.glob("*.json"):
        f.unlink(missing_ok=True)

def cache_info() -> dict:
    files = list(CACHE_DIR.glob("*.json"))
    return {
        "cache_dir": str(CACHE_DIR),
        "files":     len(files),
        "total_mb":  sum(f.stat().st_size for f in files) / 1e6,
    }
