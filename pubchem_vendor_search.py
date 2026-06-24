"""
pubchem_vendor_search.py — PubChem Fragment Vendor Search
==========================================================
Logic:
  1. Take fragment SMILES (with [*] attachment) → strip [*] → search PubChem
  2. PubChem 2D Tanimoto similarity search (threshold configurable, default 70%)
  3. For each hit → fetch properties + vendor/supplier list
  4. Return ranked list with: SMILES, MW, name, vendors, catalog URLs

Works on:
  ✅ Local version (full network access)
  ✅ Google Colab
  ❌ Streamlit Cloud (pubchem.ncbi.nlm.nih.gov blocked by egress policy)

Vendor data source:
  PubChem PUG-REST: /compound/cid/{cid}/property/...
  PubChem PUG-VIEW: /data/compound/{cid}/JSON?heading=Chemical+Vendors
  Vendor section: SourceName, RegistryID, URL
  Major vendors indexed: Sigma-Aldrich, Fluorochem, Combi-Blocks,
    Enamine, AstaTech, Matrix Scientific, TCI, Oakwood, Ambeed, BLD Pharmatech
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ─── Constants ────────────────────────────────────────────────────────────────
PUBCHEM_BASE    = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_VIEW    = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"
PUBCHEM_TIMEOUT = 20          # seconds per request
RATE_LIMIT_S    = 0.25        # 4 requests/sec max (PubChem policy)
MAX_RECORDS     = 50          # hits per similarity search
DEFAULT_THRESH  = 70          # Tanimoto %

# Known high-quality fragment suppliers (highlight in UI)
PREFERRED_VENDORS = {
    "Sigma-Aldrich", "MilliporeSigma", "TCI", "Fluorochem", "Combi-Blocks",
    "Enamine", "AstaTech", "Matrix Scientific", "Oakwood Chemical",
    "Ambeed", "BLD Pharmatech", "Chemspace", "MolPort", "Acros Organics",
    "Alfa Aesar", "Strem Chemicals",
}


def _get(url: str, retries: int = 2) -> Optional[dict]:
    """HTTP GET with retry, returns parsed JSON or None."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "AnalogBuilder/2.0 (research; mailto:kowith@ccs.tsukuba.ac.jp)",
                    "Accept": "application/json",
                }
            )
            with urllib.request.urlopen(req, timeout=PUBCHEM_TIMEOUT) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0)
            else:
                return None
    return None


def strip_attachment(frag_smiles: str) -> str:
    """
    Remove [*] dummy atom from fragment SMILES for PubChem search.
    '[*]c1ccccc1' → 'c1ccccc1'
    '[*]C(=O)N1CCCC1' → 'O=C1CCCN1'  (RDKit canonicalizes after removal)
    """
    smi = frag_smiles.strip()
    # Replace [*] with H then canonicalize
    smi_h = smi.replace("[*]", "[H]")
    mol = Chem.MolFromSmiles(smi_h)
    if mol is None:
        # Fallback: just remove [*]
        smi_h = smi.replace("[*]", "")
        mol = Chem.MolFromSmiles(smi_h)
    if mol is None:
        return smi.replace("[*]", "")
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


def similarity_search_cids(
    smiles: str,
    threshold: int = DEFAULT_THRESH,
    max_records: int = MAX_RECORDS,
) -> List[int]:
    """
    PubChem 2D Tanimoto similarity search.
    Returns list of CIDs sorted by similarity (most similar first).
    """
    encoded = urllib.parse.quote(smiles)
    url = (
        f"{PUBCHEM_BASE}/compound/fastsimilarity_2d/smiles/"
        f"{encoded}/cids/JSON"
        f"?Threshold={threshold}&MaxRecords={max_records}&StripHydrogen=true"
    )
    time.sleep(RATE_LIMIT_S)
    data = _get(url)
    if data is None:
        return []
    return data.get("IdentifierList", {}).get("CID", [])


def fetch_properties(cids: List[int]) -> List[dict]:
    """
    Batch fetch compound properties for a list of CIDs.
    Returns list of property dicts.
    """
    if not cids:
        return []
    cid_str = ",".join(str(c) for c in cids[:100])
    url = (
        f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/"
        "IsomericSMILES,MolecularFormula,MolecularWeight,"
        "IUPACName,XLogP,HBondDonorCount,HBondAcceptorCount,"
        "RotatableBondCount,ExactMass/JSON"
    )
    time.sleep(RATE_LIMIT_S)
    data = _get(url)
    if data is None:
        return []
    return data.get("PropertyTable", {}).get("Properties", [])


def fetch_vendors(cid: int) -> List[dict]:
    """
    Fetch vendor/supplier list for a compound from PubChem PUG-VIEW.
    Returns list of: {vendor, catalog_id, url, is_preferred}
    """
    url = (
        f"{PUBCHEM_VIEW}/data/compound/{cid}/JSON"
        "?heading=Chemical+Vendors"
    )
    time.sleep(RATE_LIMIT_S)
    data = _get(url)
    if data is None:
        return []

    vendors = []
    try:
        sections = data.get("Record", {}).get("Section", [])
        for section in sections:
            if "vendor" not in section.get("TOCHeading", "").lower():
                continue
            for subsec in section.get("Section", []):
                for info in subsec.get("Information", []):
                    src = info.get("Reference", [{}])[0] if info.get("Reference") else {}
                    name = src.get("SourceName", "")
                    reg_id = src.get("RegistryID", "")
                    url_v  = src.get("URL", "")
                    if name:
                        vendors.append({
                            "vendor":       name,
                            "catalog_id":   reg_id,
                            "url":          url_v,
                            "is_preferred": name in PREFERRED_VENDORS,
                        })
    except Exception:
        pass

    # Sort: preferred first
    vendors.sort(key=lambda v: (0 if v["is_preferred"] else 1, v["vendor"]))
    return vendors[:20]  # cap at 20 vendors per compound


def ro3_filter(props: dict) -> bool:
    """
    Rule-of-3 filter for fragments:
    MW ≤ 300, HBD ≤ 3, HBA ≤ 3, cLogP ≤ 3
    """
    try:
        mw  = float(props.get("MolecularWeight", 999))
        hbd = int(props.get("HBondDonorCount", 99))
        hba = int(props.get("HBondAcceptorCount", 99))
        xlogp = float(props.get("XLogP") or 99)
        return mw <= 300 and hbd <= 3 and hba <= 3 and xlogp <= 3
    except Exception:
        return False


def search_fragments_with_vendors(
    frag_smiles: str,
    threshold: int = DEFAULT_THRESH,
    max_results: int = 20,
    apply_ro3: bool = True,
    fetch_vendor_info: bool = True,
    progress_cb=None,
) -> List[dict]:
    """
    Main entry point.

    Parameters
    ----------
    frag_smiles : str
        Fragment SMILES with [*] attachment point (e.g. '[*]c1ccncc1')
    threshold : int
        Tanimoto similarity threshold 0–100 (default 70)
    max_results : int
        Max compounds to return
    apply_ro3 : bool
        Apply Rule-of-3 filter (MW≤300, HBD≤3, HBA≤3, XLogP≤3)
    fetch_vendor_info : bool
        Fetch vendor catalog info (slower, ~0.25s per compound)
    progress_cb : callable(done, total) or None

    Returns
    -------
    List of dicts:
        cid, smiles, name, mw, formula, xlogp, hbd, hba,
        vendors (list of {vendor, catalog_id, url, is_preferred}),
        pubchem_url, n_vendors
    """
    # 1. Strip attachment, clean SMILES
    search_smi = strip_attachment(frag_smiles)
    if not search_smi:
        return []

    # Validate
    mol = Chem.MolFromSmiles(search_smi)
    if mol is None:
        return []

    # 2. Similarity search
    cids = similarity_search_cids(search_smi, threshold=threshold, max_records=max_results * 3)
    if not cids:
        return []

    # 3. Fetch properties (batch)
    props_list = fetch_properties(cids[:100])
    if not props_list:
        return []

    # 4. Filter + score
    results = []
    total = len(props_list)
    for i, props in enumerate(props_list):
        if progress_cb:
            progress_cb(i + 1, total)

        if apply_ro3 and not ro3_filter(props):
            continue

        cid  = props.get("CID")
        smiles = props.get("IsomericSMILES", "")
        name   = props.get("IUPACName", f"CID {cid}")
        mw     = props.get("MolecularWeight", 0)

        # Reattach [*] to the returned SMILES for use in analog generation
        frag_smi_out = smiles  # raw PubChem SMILES (no [*] yet)

        entry = {
            "cid":         cid,
            "smiles":      smiles,          # canonical PubChem SMILES
            "frag_smiles": f"[*]{smiles}",  # with attachment point (simplified)
            "name":        name[:60],
            "mw":          float(mw) if mw else 0.0,
            "formula":     props.get("MolecularFormula", ""),
            "xlogp":       props.get("XLogP"),
            "hbd":         props.get("HBondDonorCount", 0),
            "hba":         props.get("HBondAcceptorCount", 0),
            "rot":         props.get("RotatableBondCount", 0),
            "vendors":     [],
            "n_vendors":   0,
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
        }

        # 5. Fetch vendor info per compound
        if fetch_vendor_info and cid:
            vendors = fetch_vendors(cid)
            entry["vendors"]   = vendors
            entry["n_vendors"] = len(vendors)

        results.append(entry)

        if len(results) >= max_results:
            break

    # Sort by: preferred vendors first, then MW
    results.sort(key=lambda r: (-r["n_vendors"], r["mw"]))
    return results


def format_vendor_table(results: List[dict]) -> list:
    """
    Flatten results into rows for Streamlit dataframe display.
    Each row = one compound (with top vendor shown).
    """
    rows = []
    for r in results:
        top_vendor = r["vendors"][0] if r["vendors"] else {}
        rows.append({
            "CID":          r["cid"],
            "SMILES":       r["smiles"],
            "Name":         r["name"],
            "MW":           r["mw"],
            "LogP":         r["xlogp"],
            "HBD/HBA":      f"{r['hbd']}/{r['hba']}",
            "Vendors":      r["n_vendors"],
            "Top vendor":   top_vendor.get("vendor", "—"),
            "Catalog ID":   top_vendor.get("catalog_id", "—"),
            "PubChem":      r["pubchem_url"],
            "Vendor URL":   top_vendor.get("url", ""),
            "✓ Preferred":  "✅" if top_vendor.get("is_preferred") else "",
        })
    return rows


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing PubChem vendor search...")
    results = search_fragments_with_vendors(
        frag_smiles="[*]c1ccncc1",   # 4-pyridinyl
        threshold=80,
        max_results=5,
        apply_ro3=True,
        fetch_vendor_info=True,
    )
    print(f"\nFound {len(results)} fragments:")
    for r in results:
        print(f"  CID {r['cid']}: {r['smiles']} | MW={r['mw']} | {r['n_vendors']} vendors")
        for v in r['vendors'][:2]:
            marker = "⭐" if v['is_preferred'] else "  "
            print(f"    {marker} {v['vendor']} — {v['catalog_id']}")
            if v['url']: print(f"       {v['url']}")
