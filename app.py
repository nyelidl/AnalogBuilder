"""
app.py — ⌬+⌬ Analog Builder · Streamlit web application
Redesigned UX: warm tones, student-friendly, two separate tracks,
sidebar progress indicator, advanced options collapsed.
"""

from __future__ import annotations

import base64
import io
import json
import os
import shlex
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

import core

# Optional in-browser molecule sketcher
try:
    from streamlit_ketcher import st_ketcher
    _KETCHER_OK = True
except Exception:
    _KETCHER_OK = False

# Logo
LOGO_URL = "https://raw.githubusercontent.com/nyelidl/AnalogBuilder/main/.fig/AB.svg"
LB_URL   = "https://raw.githubusercontent.com/nyelidl/AnalogBuilder/main/.fig/LB.svg"
SB_URL   = "https://raw.githubusercontent.com/nyelidl/AnalogBuilder/main/.fig/SB.svg"

# ─────────────────────────────────────────────────────────────────────────────
# Page config + global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="⌬+⌬ Analog Builder",
    page_icon=LOGO_URL,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Warm base ── */
[data-testid="stAppViewContainer"] {
    background: #FAF7F2;
}
[data-testid="stSidebar"] {
    background: #F0EAE0;
    border-right: 1px solid #E0D6C8;
}

/* ── Typography ── */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #2C2C2C;
}
h1 { font-size: 1.6rem !important; font-weight: 700; color: #2C2C2C; }
h2 { font-size: 1.2rem !important; font-weight: 600; color: #2C2C2C; }
h3 { font-size: 1.0rem !important; font-weight: 600; color: #3D7A74; }

/* ── Primary button: amber ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #E8A020 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.4rem !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #C88010 !important;
}

/* ── Secondary button ── */
[data-testid="stButton"] > button {
    border-radius: 8px !important;
    border: 1px solid #C8B89A !important;
    background: #FAF7F2 !important;
    color: #2C2C2C !important;
}

/* ── Mode cards on landing ── */
.mode-card {
    background: #FFFFFF;
    border: 2px solid #E0D6C8;
    border-radius: 14px;
    padding: 2rem 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.mode-card:hover {
    border-color: #E8A020;
    box-shadow: 0 4px 16px rgba(232,160,32,0.15);
}
.mode-card .icon { font-size: 3rem; margin-bottom: 0.5rem; }
.mode-card h2 { color: #2C2C2C; margin: 0.4rem 0 0.6rem; }
.mode-card p { color: #6B5E4E; font-size: 0.9rem; line-height: 1.5; margin: 0; }

/* ── Step progress in sidebar ── */
.step-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border-radius: 8px;
    margin-bottom: 4px;
    font-size: 0.88rem;
    color: #6B5E4E;
    cursor: pointer;
}
.step-item.active {
    background: #E8A020;
    color: #fff;
    font-weight: 600;
}
.step-item.done {
    color: #3D7A74;
    font-weight: 500;
}
.step-dot {
    width: 22px; height: 22px;
    border-radius: 50%;
    background: #E0D6C8;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-weight: 700; flex-shrink: 0;
    color: #6B5E4E;
}
.step-item.active .step-dot { background: rgba(255,255,255,0.3); color: #fff; }
.step-item.done .step-dot { background: #3D7A74; color: #fff; }

/* ── Hint text ── */
.hint { color: #8B7355; font-size: 0.82rem; margin-top: -6px; margin-bottom: 10px; }

/* ── Info cards ── */
.info-card {
    background: #FFF8EE;
    border-left: 3px solid #E8A020;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0 1rem;
    font-size: 0.88rem;
    color: #5A4A35;
}

/* ── Metric row ── */
.metric-row {
    display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0;
}
.metric-box {
    background: #fff;
    border: 1px solid #E0D6C8;
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    min-width: 120px;
    text-align: center;
}
.metric-box .val { font-size: 1.5rem; font-weight: 700; color: #E8A020; }
.metric-box .lbl { font-size: 0.78rem; color: #8B7355; margin-top: 2px; }

/* ── Dataframe tweaks ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Expander (Advanced) ── */
[data-testid="stExpander"] summary {
    font-size: 0.85rem;
    color: #8B7355;
}

/* ── Fixed page footer ── */
.page-footer {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #F0EAE0;
    border-top: 1px solid #E0D6C8;
    padding: 6px 20px;
    font-size: 0.75rem;
    color: #A89070;
    z-index: 999;
    display: flex;
    align-items: center;
    gap: 6px;
}
.page-footer a { color: #A89070; text-decoration: none; }
.page-footer a:hover { color: #E8A020; }

/* ── Mode tab bar (first two buttons under the footer markup) ── */
/* Tabs are the two columns at the very top of the main area.
   We style them to read as underline tabs rather than filled buttons. */
.main .block-container div[data-testid="stHorizontalBlock"]:first-of-type
    button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
    color: #8B7355 !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 0.2rem !important;
    box-shadow: none !important;
}
.main .block-container div[data-testid="stHorizontalBlock"]:first-of-type
    button:hover {
    color: #2C2C2C !important;
    background: transparent !important;
}
/* active tab = the primary-typed button */
.main .block-container div[data-testid="stHorizontalBlock"]:first-of-type
    button[kind="primary"] {
    background: transparent !important;
    color: #2C2C2C !important;
    border-bottom: 3px solid #E8A020 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="page-footer">' +
    '⌬+⌬ Analog Builder &nbsp;—&nbsp;' +
    '<a href="mailto:kowith@ccs.tsukuba.ac.jp">kowith@ccs.tsukuba.ac.jp</a>' +
    '</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    # Navigation
    "mode": None,               # "ligand" | "structure"
    "step": 1,                  # current step within a track
    # Step 1
    "parent_smiles": "",
    "parent_name": "compound",
    "parent_mol": None,
    # Step 1B (structure only)
    "receptor_path": None,
    "protein_path": None,
    "complex_path": None,
    "ref_ligand_path": None,
    # Step 2
    "selected_atoms": set(),
    "concerted": False,
    "allow_heteroatom_H": False,
    # Step 3 – quick (always visible)
    "risk": "Moderate",
    "n_analogs": 20,
    "rank_by": "Overall drug-likeness (recommended)",
    "rank_code": "Balanced (100-pt weights)",
    # Step 3 – advanced (collapsed)
    "weights": {"potency": 30, "selectivity": 10, "solubility": 25,
                "metabolic": 15, "synthesis": 10, "novelty": 10},
    "categories_on": {k: True for k in core.CATEGORY_BASE_GOALS},
    "max_MW": 600.0,
    "avoid_nitro": True,
    "avoid_aldehyde": True,
    "avoid_reactive": True,
    "avoid_toxic": True,
    "custom_frags_text": "",
    # Step 3C pocket guidance (structure track)
    "pocket_residue_text": "",
    "accept_pocket_suggestions": True,
    "max_pocket_frags": 6,
    "pocket_frags": [],
    # Step 4
    "analogs_df": None,
    # Step 5 docking
    "docking_ligands": None,
    "docking_summary": None,
    "cifp_results": None,
    "work_dir": None,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def get_work_dir() -> Path:
    if st.session_state.work_dir is None:
        st.session_state.work_dir = Path(tempfile.mkdtemp(prefix="analog_"))
    return Path(st.session_state.work_dir)


def go(step: int):
    st.session_state.step = step
    st.rerun()


def hint(text: str):
    st.markdown(f'<p class="hint">💡 {text}</p>', unsafe_allow_html=True)


def info_card(text: str):
    st.markdown(f'<div class="info-card">{text}</div>', unsafe_allow_html=True)


def svg_img(mol, highlight=None, size=(500, 380)):
    """Generate SVG as base64 img tag (works locally, may be stripped on Cloud)."""
    if mol is None:
        return ""
    svg = core.draw_mol_svg(mol, highlight=list(highlight or []), size=size)
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" style="max-width:100%;border-radius:10px;border:1px solid #E0D6C8;">'


def show_mol(mol, highlight=None, size=(400, 300), use_container_width=True):
    """Render a molecule using st.image (PNG) — works reliably on Streamlit Cloud.
    Uses MolsToGridImage with returnPNG=True which is confirmed working."""
    if mol is None:
        return
    try:
        AllChem.Compute2DCoords(mol)
        png = Draw.MolsToGridImage(
            [mol], molsPerRow=1,
            subImgSize=size,
            highlightAtomLists=[list(highlight or [])],
            returnPNG=True,
        )
        st.image(png, use_container_width=use_container_width)
    except Exception:
        st.caption("(Could not render structure)")


def _render_step1_receptor_and_continue(smiles: str, name: str):
    """Shared Step-1 footer: receptor loader (structure mode) + continue button."""
    md = st.session_state.mode

    if md == "structure":
        st.markdown("### Protein receptor")
        info_card("The receptor is the protein your compound binds to. "
                  "Search by name, enter a PDB ID, or upload a file.")
        rec_src = st.radio(
            "Load receptor from",
            ["🔍 Search RCSB", "#️⃣ PDB ID", "📁 Upload file"],
            horizontal=True,
        )

        if rec_src == "🔍 Search RCSB":
            rcsb_query = st.text_input(
                "Search RCSB PDB",
                value="",
                placeholder="e.g. EGFR kinase, JAK2, insulin receptor",
                key="rcsb_search_q",
            )
            hint("Search by protein name, gene, UniProt ID, or keyword.")

            if st.button("Search RCSB", key="rcsb_search_btn") and rcsb_query.strip():
                with st.spinner("Searching RCSB PDB…"):
                    st.session_state["_rcsb_results"] = core.search_rcsb(rcsb_query.strip(), max_results=8)

            rcsb_results = st.session_state.get("_rcsb_results", [])
            if rcsb_results:
                st.markdown(f"**{len(rcsb_results)} results**")
                for r in rcsb_results:
                    cols = st.columns([1, 6, 2])
                    with cols[0]:
                        st.markdown(f"**{r['id']}**")
                    with cols[1]:
                        st.caption(f"{r['title']}")
                        st.caption(f"{r['resolution']} · {r['method']} · {r['organism']}")
                    with cols[2]:
                        if st.button("Use", key=f"rcsb_use_{r['id']}"):
                            with st.spinner(f"Downloading {r['id']}…"):
                                try:
                                    work = get_work_dir()
                                    path = core.download_pdb(r["id"], work)
                                    prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                                    st.session_state.receptor_path   = path
                                    st.session_state.protein_path    = prot
                                    st.session_state.complex_path    = path
                                    st.session_state.ref_ligand_path = lig
                                    st.success(f"Receptor loaded ({r['id']}) ✅")
                                except Exception as e:
                                    st.error(f"Could not download: {e}")

        elif rec_src == "#️⃣ PDB ID":
            pdb_id = st.text_input("4-letter PDB ID", value="", max_chars=4, placeholder="e.g. 1M17")
            hint("Example: 1M17 is EGFR, 6VXX is SARS-CoV-2 spike.")
            if st.button("Load receptor", key="load_rec") and pdb_id.strip():
                with st.spinner("Downloading from RCSB…"):
                    try:
                        work = get_work_dir()
                        path = core.download_pdb(pdb_id.strip().upper(), work)
                        prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                        st.session_state.receptor_path   = path
                        st.session_state.protein_path    = prot
                        st.session_state.complex_path    = path
                        st.session_state.ref_ligand_path = lig
                        st.success(f"Receptor loaded ({pdb_id.upper()}) ✅")
                    except Exception as e:
                        st.error(f"Could not download: {e}")

        else:  # Upload file
            up = st.file_uploader("Upload .pdb or .cif file", type=["pdb", "cif"])
            if up:
                work = get_work_dir()
                raw = work / up.name
                raw.write_bytes(up.read())
                try:
                    path = core.cif_to_pdb_if_needed(str(raw))
                    prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                    st.session_state.receptor_path   = path
                    st.session_state.protein_path    = prot
                    st.session_state.complex_path    = path
                    st.session_state.ref_ligand_path = lig
                    st.success("Receptor uploaded ✅")
                except Exception as e:
                    st.error(f"Could not process file: {e}")

        if st.session_state.receptor_path:
            st.success(f"✅ Receptor ready: `{Path(st.session_state.receptor_path).name}`")
            if st.session_state.ref_ligand_path:
                st.info("Co-crystal ligand detected — will be used as reference pose")

    st.write("")
    if md == "structure" and not st.session_state.receptor_path:
        st.caption("⚠️ Load a receptor above before continuing.")

    if st.button("Load compound & continue →", type="primary", disabled=not bool(smiles and smiles.strip())):
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            st.error("That SMILES doesn't look right. Check for typos and try again.")
        elif md == "structure" and not st.session_state.receptor_path:
            st.error("Please load a receptor before continuing.")
        else:
            AllChem.Compute2DCoords(mol)
            st.session_state.parent_smiles  = smiles.strip()
            st.session_state.parent_name    = name.strip() or "compound"
            st.session_state.parent_mol     = mol
            st.session_state.selected_atoms = set()
            st.session_state.analogs_df     = None
            go(2)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – progress tracker
# ─────────────────────────────────────────────────────────────────────────────

LIGAND_STEPS  = ["Parent compound", "Choose atoms", "Design options", "View results", "Docking", "Export"]
STRUCT_STEPS  = ["Parent + receptor", "Choose atoms", "Pocket guidance", "View results", "Docking & cIFP", "Export"]

def render_sidebar():
    st.sidebar.markdown(
        '<style>[data-testid="stSidebar"] [data-testid="stImage"] {text-align: center;}</style>',
        unsafe_allow_html=True,
    )
    st.sidebar.image(LOGO_URL, width=160)
    st.sidebar.markdown(
        '<p style="text-align:center;font-size:0.78rem;color:#8B7355;margin-top:-8px;">Ligand design for everyone</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    mode = st.session_state.mode
    if mode is None:
        st.sidebar.markdown('<p style="color:#8B7355;font-size:0.85rem;">Select a mode to begin.</p>',
                            unsafe_allow_html=True)
        return

    steps = LIGAND_STEPS if mode == "ligand" else STRUCT_STEPS
    current = st.session_state.step

    _mode_icon = LB_URL if mode == "ligand" else SB_URL
    _mode_name = "Ligand-based" if mode == "ligand" else "Structure-based"
    st.sidebar.markdown(
        f'<p style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;'
        f'color:#8B7355;margin-bottom:8px;">'
        f'<img src="{_mode_icon}" hight="100" style="vertical-align:middle;margin-right:4px;"/>'
        f'{_mode_name} track</p>',
        unsafe_allow_html=True
    )

    for i, label in enumerate(steps, start=1):
        cls = "active" if i == current else ("done" if i < current else "")
        if st.sidebar.button(
            f"{'●' if i == current else ('✓' if i < current else str(i))}  {label}",
            key=f"nav_{i}",
            use_container_width=True,
            type="primary" if i == current else "secondary",
        ):
            if i < current or (i == current + 1 and _step_complete(current)):
                go(i)

    st.sidebar.divider()
    if st.sidebar.button("↩ Change mode", use_container_width=True):
        st.session_state.mode = None
        st.session_state.step = 1
        st.session_state.parent_mol = None
        st.session_state.analogs_df = None
        st.rerun()

    st.sidebar.caption(f"Fragment library: {len(core.BUILTIN_LIBRARY)} groups")
    st.sidebar.divider()
    st.sidebar.markdown(
        '<p style="font-size:0.75rem;color:#A89070;line-height:1.5;margin:0;">'
        '⌬+⌬ Analog Builder<br>'
        '<a href="mailto:kowith@ccs.tsukuba.ac.jp" style="color:#A89070;">kowith@ccs.tsukuba.ac.jp</a>'
        '</p>',
        unsafe_allow_html=True,
    )


def _step_complete(step: int) -> bool:
    if step == 1:
        return st.session_state.parent_mol is not None
    if step == 2:
        return len(st.session_state.selected_atoms) > 0
    if step == 3:
        return True
    if step == 4:
        return st.session_state.analogs_df is not None
    return True


render_sidebar()


# ─────────────────────────────────────────────────────────────────────────────
# LANDING – mode picker
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.mode is None:
    st.markdown(
        f'<div style="text-align:center;margin:40px 0 8px;">'
        f'<img src="{LOGO_URL}" width="260" style="display:inline-block;"/>'
        f'</div>'
        f'<p style="text-align:center;color:#8B7355;margin-bottom:28px;">'
        f'Design new drug candidates by modifying a parent compound. Choose how you want to work:</p>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown(f"""
        <div class="mode-card">
            <div class="icon"><img src="{LB_URL}" hight="100" style="display:inline-block;"/></div>
            <h2>Ligand-based</h2>
            <p>Start with just a SMILES string.<br>
            Great for exploring substitutions quickly — no protein structure needed.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Start ligand-based →", type="primary", use_container_width=True, key="pick_ligand"):
            st.session_state.mode = "ligand"
            st.session_state.step = 1
            st.rerun()

    with col_r:
        st.markdown(f"""
        <div class="mode-card">
            <div class="icon"><img src="{SB_URL}" hight="100" style="display:inline-block;"/></div>
            <h2>Structure-based</h2>
            <p>Upload or fetch a protein structure.<br>
            Analogs are guided by the actual binding pocket environment.</p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        if st.button("Start structure-based →", type="primary", use_container_width=True, key="pick_struct"):
            st.session_state.mode = "structure"
            st.session_state.step = 1
            st.rerun()

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Mode tab bar  (switch tracks anytime after picking)
# ─────────────────────────────────────────────────────────────────────────────

mode  = st.session_state.mode
step  = st.session_state.step

def switch_mode(new_mode: str):
    """Switch track. Keeps the parent compound + atom selection so the user
    doesn't lose work, but resets downstream results that are mode-specific."""
    if new_mode == st.session_state.mode:
        return
    st.session_state.mode = new_mode
    # Keep step if it still exists in the new track, else clamp.
    new_len = len(LIGAND_STEPS if new_mode == "ligand" else STRUCT_STEPS)
    st.session_state.step = min(st.session_state.step, new_len)
    st.rerun()

tab_l, tab_r = st.columns(2)
with tab_l:
    st.markdown(
        f'<div style="text-align:center;margin-bottom:-8px;">'
        f'<img src="{LB_URL}" hight="100" style="opacity:{1.0 if mode=="ligand" else 0.4};"/></div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Ligand-based",
        key="tab_ligand",
        use_container_width=True,
        type="primary" if mode == "ligand" else "secondary",
    ):
        switch_mode("ligand")
with tab_r:
    st.markdown(
        f'<div style="text-align:center;margin-bottom:-8px;">'
        f'<img src="{SB_URL}" width="70" style="opacity:{1.0 if mode=="structure" else 0.4};"/></div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Structure-based",
        key="tab_structure",
        use_container_width=True,
        type="primary" if mode == "structure" else "secondary",
    ):
        switch_mode("structure")

st.markdown('<hr style="margin:0.2rem 0 1.2rem 0;border:none;border-top:1px solid #E0D6C8;">',
            unsafe_allow_html=True)

# Re-read in case mode changed above
mode  = st.session_state.mode
step  = st.session_state.step
steps = LIGAND_STEPS if mode == "ligand" else STRUCT_STEPS

# Step breadcrumb
st.markdown(
    f'<p style="font-size:0.8rem;color:#8B7355;margin-bottom:0;">'
    f'Step {step} of {len(steps)}: <strong>{steps[step-1]}</strong></p>',
    unsafe_allow_html=True
)
st.markdown(f"## {steps[step-1]}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Parent compound  (both tracks)
# ─────────────────────────────────────────────────────────────────────────────

if step == 1:
    # ── Input-mode toggle ────────────────────────────────────────────────────
    input_options = ["🔍 Search PubChem", "⌨️ Paste SMILES", "✏️ Draw it"]
    input_tab = st.radio(
        "How do you want to enter your compound?",
        input_options,
        horizontal=True,
        label_visibility="collapsed",
    )

    draw_mode  = (input_tab == "✏️ Draw it") and _KETCHER_OK
    paste_mode = (input_tab == "⌨️ Paste SMILES") or (input_tab == "✏️ Draw it" and not _KETCHER_OK)
    pubchem_mode = (input_tab == "🔍 Search PubChem")

    # ── PUBCHEM SEARCH ───────────────────────────────────────────────────────
    if pubchem_mode:
        st.markdown("#### 🔍 Search compound from PubChem")
        pc_col1, pc_col2 = st.columns([5, 1])
        with pc_col1:
            pc_query = st.text_input(
                "Compound name",
                placeholder="e.g. imatinib, apigenin, caffeine, aspirin…",
                key="pubchem_query",
            )
        with pc_col2:
            st.markdown("<div style='height:1.75rem;'></div>", unsafe_allow_html=True)
            pc_search = st.button("Search", key="pc_search_btn", type="secondary")

        if pc_search and pc_query.strip():
            with st.spinner(f"Searching PubChem for '{pc_query.strip()}'…"):
                _sr = core.search_pubchem(pc_query.strip())
                st.session_state["_pc_result"] = _sr
                if _sr.get("found") and (_sr.get("smiles") or "").strip():
                    st.session_state.parent_smiles = _sr["smiles"]
                    st.session_state.parent_name = (
                        (_sr["iupac"] or pc_query)[:20].lower().replace(" ", "_")
                    )

        # Show result
        _sr = st.session_state.get("_pc_result")
        if _sr and _sr.get("found"):
            _ic, _imgc = st.columns([3, 1])
            with _ic:
                st.markdown(
                    f"**{_sr['iupac']}**  \n"
                    f"`{_sr['formula']}` · {_sr['mw']:.2f} g/mol · "
                    f"[PubChem CID {_sr['cid']}]({_sr['url']})"
                )
            with _imgc:
                st.image(_sr["img_url"], width=100)
            if not (_sr.get("smiles") or "").strip():
                st.warning("This PubChem result did not return a usable SMILES string.")
        elif _sr and not _sr.get("found"):
            st.error(f"Not found: {_sr.get('error', 'Unknown error')}")

        # SMILES text input — auto-filled from PubChem search
        smiles = st.text_input(
            "SMILES string",
            value=st.session_state.parent_smiles,
            key="smiles_in_pc",
            help="Auto-filled from PubChem search, or paste your own SMILES here.",
        )
        st.session_state.parent_smiles = smiles

        name = st.text_input(
            "Compound name",
            value=st.session_state.parent_name,
            key="pc_compound_name",
        )
        hint("Used to label your output files.")

        _render_step1_receptor_and_continue(smiles, name)

    # ── DRAW MODE: full-width sketcher, preview + form below ─────────────────
    elif draw_mode:
        hint("Draw your molecule, then click **Apply** in the sketcher to capture it.")
        drawn = st_ketcher(
            st.session_state.parent_smiles or "",
            key="ketcher_draw",
            height=480,
        )
        smiles = drawn or st.session_state.parent_smiles

        prev_col, form_col = st.columns([1, 1], gap="large")
        with prev_col:
            mol_preview = Chem.MolFromSmiles(smiles.strip()) if smiles and smiles.strip() else None
            if mol_preview:
                c_sites = core.attachable_atom_indices(mol_preview, carbon_only=True)
                st.markdown("**Preview** — highlighted atoms can be modified")
                show_mol(mol_preview, highlight=c_sites)
                st.caption(f"Captured SMILES: `{smiles}`  ·  {mol_preview.GetNumAtoms()} atoms · {len(c_sites)} modifiable C–H sites")
            else:
                st.markdown(
                    '<div style="background:#F0EAE0;border-radius:12px;height:240px;'
                    'display:flex;align-items:center;justify-content:center;color:#A89070;">'
                    '<span style="font-size:0.9rem;">Draw a molecule and click Apply to see the preview</span></div>',
                    unsafe_allow_html=True
                )
        with form_col:
            name = st.text_input("Give it a short name", value=st.session_state.parent_name, placeholder="e.g. compound_1")
            hint("Used to label your output files.")
            _render_step1_receptor_and_continue(smiles, name)

    # ── PASTE MODE (or sketcher unavailable): side-by-side ───────────────────
    else:
        col_form, col_mol = st.columns([1, 1], gap="large")

        with col_form:
            if input_tab == "✏️ Draw it" and not _KETCHER_OK:
                st.warning(
                    "The drawing tool isn't installed here. Add `streamlit-ketcher` to "
                    "requirements.txt to enable it. For now, paste a SMILES instead."
                )
            smiles = st.text_area(
                "Paste your compound SMILES",
                value=st.session_state.parent_smiles,
                height=90,
                placeholder="e.g. CC1=CC=CC=C1",
            )
            hint("SMILES is a text code for a molecule. Copy it from ChemDraw, PubChem, or any chemistry database.")

            name = st.text_input("Give it a short name", value=st.session_state.parent_name, placeholder="e.g. compound_1")
            hint("Used to label your output files.")
            _render_step1_receptor_and_continue(smiles, name)

        with col_mol:
            mol_preview = Chem.MolFromSmiles(smiles.strip()) if smiles.strip() else None
            if mol_preview:
                c_sites = core.attachable_atom_indices(mol_preview, carbon_only=True)
                st.markdown("**Preview** — highlighted atoms can be modified")
                show_mol(mol_preview, highlight=c_sites)
                st.caption(f"{mol_preview.GetNumAtoms()} atoms · {len(c_sites)} modifiable C–H sites")
            else:
                st.markdown(
                    '<div style="background:#F0EAE0;border-radius:12px;height:320px;'
                    'display:flex;align-items:center;justify-content:center;color:#A89070;">'
                    '<span style="font-size:0.9rem;">Molecule preview appears here</span></div>',
                    unsafe_allow_html=True
                )

            if mode == "structure" and st.session_state.receptor_path:
                st.markdown("**Receptor status**")
                st.success(f"✅ {Path(st.session_state.receptor_path).name}")
                if st.session_state.ref_ligand_path:
                    st.info("Co-crystal ligand detected — will be used as reference pose")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Choose atoms
# ─────────────────────────────────────────────────────────────────────────────

elif step == 2:
    mol = st.session_state.parent_mol
    if mol is None:
        st.warning("Go back to Step 1 and load a compound first.")
        st.stop()

    attachable = core.attachable_atom_indices(mol, carbon_only=False)
    c_only     = core.attachable_atom_indices(mol, carbon_only=True)

    info_card("Click an atom index in the list to select it. "
              "Selected atoms will be highlighted on the structure — those are the spots where new groups will be added.")

    col_pick, col_view = st.columns([1, 1], gap="large")

    with col_pick:
        st.markdown("### Pick attachment points")
        hint("Stick to C–H sites (carbon atoms) unless you have a specific reason to modify N–H or O–H.")

        new_sel = set()
        for idx in attachable:
            atom  = mol.GetAtomWithIdx(idx)
            atype = "C–H" if atom.GetAtomicNum() == 6 else f"{atom.GetSymbol()}–H"
            label = f"Atom {idx}  ({atype})"
            if idx not in c_only:
                label += "  *(heteroatom)*"
            if st.checkbox(label, value=idx in st.session_state.selected_atoms, key=f"atm_{idx}"):
                new_sel.add(idx)

        st.session_state.selected_atoms = new_sel

        st.write("")
        st.markdown("### Options")
        st.session_state.allow_heteroatom_H = st.checkbox(
            "Allow N–H / O–H / S–H substitution",
            value=st.session_state.allow_heteroatom_H,
        )
        hint("By default only carbon (C–H) sites are modified. Turn this on to also substitute on nitrogen, oxygen, or sulfur.")
        st.session_state.concerted = st.checkbox(
            "Concerted mode — attach the same group to all selected atoms at once",
            value=st.session_state.concerted,
        )
        hint("Off: each analog changes one site. On: each analog changes all selected sites together (larger, multi-substituted analogs).")

    with col_view:
        st.markdown("### Structure")
        show_mol(mol, highlight=sorted(new_sel))
        if new_sel:
            st.caption(f"Selected: atoms {sorted(new_sel)}")
        else:
            st.caption("No atoms selected yet")

    st.write("")
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            go(1)
    with col_next:
        if st.button("Continue →", type="primary", disabled=len(new_sel) == 0):
            go(3)
        if not new_sel:
            st.caption("Select at least one atom to continue.")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Design options  (ligand track)
# ─────────────────────────────────────────────────────────────────────────────

elif step == 3 and mode == "ligand":
    info_card("These settings control what kinds of groups are added and how the results are ranked. "
              "The defaults work well — you can leave them and click Generate.")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown("### How many analogs?")
        st.session_state.n_analogs = st.slider(
            "Number of analogs to generate", 5, 20,
            min(st.session_state.n_analogs, 20), step=1,
        )
        hint("Up to 20 analogs are generated. Fewer is faster to review.")

        st.markdown("### How adventurous?")
        risk_map = {
            "Conservative — small groups only": "Conservative",
            "Moderate — balanced (recommended)": "Moderate",
            "Exploratory — larger groups allowed": "Exploratory",
        }
        risk_label = st.radio(
            "Substitution size",
            list(risk_map.keys()),
            index=list(risk_map.values()).index(st.session_state.risk),
        )
        st.session_state.risk = risk_map[risk_label]
        hint("Conservative keeps modifications small and drug-like. Exploratory allows bigger substituents.")

    with col_b:
        st.markdown("### Rank results by")
        rank_map = {
            "Overall drug-likeness (recommended)": "Balanced (100-pt weights)",
            "Most similar to parent": "Similarity to parent",
            "Best predicted solubility": "Solubility (ESOL)",
            "Easiest to synthesise": "Synthetic feasibility",
        }
        rank_labels = list(rank_map.keys())
        current_label = st.session_state.rank_by if st.session_state.rank_by in rank_labels else rank_labels[0]
        rank_label = st.radio(
            "Sort analogs by",
            rank_labels,
            index=rank_labels.index(current_label),
        )
        st.session_state.rank_by = rank_label
        st.session_state.rank_code = rank_map[rank_label]

        st.markdown("### Fragment categories")
        hint("Uncheck any group types you want to exclude from the library.")
        cat_cols = st.columns(2)
        cats = list(core.CATEGORY_BASE_GOALS.keys())
        new_cats = {}
        for i, cat in enumerate(cats):
            with cat_cols[i % 2]:
                new_cats[cat] = st.checkbox(
                    cat.replace("_", " ").capitalize(),
                    value=st.session_state.categories_on.get(cat, True),
                    key=f"cat_{cat}",
                )
        st.session_state.categories_on = new_cats

    # Advanced
    with st.expander("⚙️ Advanced options"):
        adv1, adv2 = st.columns(2)
        with adv1:
            st.markdown("**Structural filters**")
            st.session_state.avoid_nitro     = st.checkbox("Remove nitro groups",   value=st.session_state.avoid_nitro)
            st.session_state.avoid_aldehyde  = st.checkbox("Remove aldehydes",      value=st.session_state.avoid_aldehyde)
            st.session_state.avoid_reactive  = st.checkbox("Remove reactive groups",value=st.session_state.avoid_reactive)
            st.session_state.avoid_toxic     = st.checkbox("Remove toxic flags",    value=st.session_state.avoid_toxic)
            st.session_state.max_MW = st.number_input("Max molecular weight (Da)", value=st.session_state.max_MW, step=25.0)

        with adv2:
            st.markdown("**Goal weights** (must sum to ~100)")
            w = st.session_state.weights
            new_w = {}
            for k in w:
                new_w[k] = st.slider(k.capitalize(), 0, 100, int(w[k]), step=5, key=f"w_{k}")
            st.session_state.weights = new_w

        st.markdown("**Custom fragments** — one SMILES with `[*]` per line")
        st.session_state.custom_frags_text = st.text_area(
            "Custom fragments", value=st.session_state.custom_frags_text,
            height=80, label_visibility="collapsed",
            placeholder="[*]C1CC1\n[*]OCC"
        )

    st.write("")
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            go(2)
    with col_next:
        if st.button("Generate analogs →", type="primary"):
            go(4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Pocket guidance  (structure track)
# ─────────────────────────────────────────────────────────────────────────────

elif step == 3 and mode == "structure":
    info_card("We'll analyse which residues are near the binding site and suggest the best functional groups to add.")

    # Auto pocket analysis
    if st.session_state.complex_path and os.path.exists(str(st.session_state.complex_path)):
        col_analysis, col_result = st.columns([1, 1], gap="large")

        with col_analysis:
            st.markdown("### Automatic pocket analysis")
            hint("Click the button to detect residues within 6 Å of the co-crystal ligand.")
            cutoff = st.slider("Pocket distance cutoff (Å)", 4.0, 10.0, 6.0, 0.5)

            if st.button("Analyse pocket", type="primary"):
                with st.spinner("Analysing binding pocket…"):
                    try:
                        pocket_df, contact_df, growth_df, lig_atoms = core.analyze_complex_distance_shell(
                            st.session_state.complex_path,
                            pocket_cutoff=cutoff,
                        )
                        residue_codes = [x for x in growth_df["aa_one"].tolist() if x]
                        st.session_state.pocket_residue_text = " ".join(
                            core.AA_ONE_TO_THREE.get(r, r) for r in residue_codes
                        )
                        active_lib = [f for f in core.BUILTIN_LIBRARY]
                        _, _, pocket_frags = core.suggest_fragments_from_residues(
                            residue_codes, active_lib, st.session_state.max_pocket_frags
                        )
                        st.session_state.pocket_frags = pocket_frags
                        st.success(
                            f"Found {len(pocket_df)} pocket residues, "
                            f"{len(growth_df)} growth opportunities"
                        )
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")

            st.markdown("### Or paste residues manually")
            hint("Type residue names like: ASP315 LYS89 TYR102 — useful if you already know the key contacts.")
            manual = st.text_input(
                "Pocket residues",
                value=st.session_state.pocket_residue_text,
                placeholder="ASP315 LYS89 TYR102",
                label_visibility="collapsed",
            )
            if manual != st.session_state.pocket_residue_text:
                st.session_state.pocket_residue_text = manual
                codes = core.parse_pocket_residues(manual)
                if codes:
                    _, _, pf = core.suggest_fragments_from_residues(codes, core.BUILTIN_LIBRARY, 6)
                    st.session_state.pocket_frags = pf

        with col_result:
            if st.session_state.pocket_frags:
                st.markdown("### Suggested functional groups")
                hint("These groups match the chemistry of your binding pocket residues.")
                fdf = pd.DataFrame([
                    {"Group": f.name, "Category": f.category, "Why": f.notes or f.category}
                    for f in st.session_state.pocket_frags
                ])
                st.dataframe(fdf, use_container_width=True, hide_index=True)
            else:
                st.markdown(
                    '<div style="background:#F0EAE0;border-radius:12px;padding:2rem;'
                    'text-align:center;color:#A89070;margin-top:1rem;">'
                    'Suggested groups will appear here after analysis</div>',
                    unsafe_allow_html=True
                )
    else:
        st.warning("No receptor file found. Go back to Step 1 and load a receptor.")

    # Shared quick settings
    st.divider()
    st.markdown("### Generation settings")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.n_analogs = st.slider("Number of analogs", 5, 20, min(st.session_state.n_analogs, 20), step=1)
    with c2:
        risk_map = {
            "Conservative": "Conservative",
            "Moderate (recommended)": "Moderate",
            "Exploratory": "Exploratory",
        }
        r = st.radio("Substitution size", list(risk_map.keys()),
                     index=list(risk_map.values()).index(st.session_state.risk), horizontal=True)
        st.session_state.risk = risk_map[r]

    with st.expander("⚙️ Advanced options"):
        st.session_state.max_MW      = st.number_input("Max MW (Da)", value=st.session_state.max_MW, step=25.0)
        st.session_state.avoid_nitro = st.checkbox("Remove nitro groups",    value=st.session_state.avoid_nitro)
        st.session_state.avoid_toxic = st.checkbox("Remove toxic flags",     value=st.session_state.avoid_toxic)
        st.session_state.max_pocket_frags = st.slider("Max pocket-guided fragments", 3, 20,
                                                       st.session_state.max_pocket_frags)

    st.write("")
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            go(2)
    with col_next:
        if st.button("Generate analogs →", type="primary"):
            go(4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – View results
# ─────────────────────────────────────────────────────────────────────────────

elif step == 4:
    mol = st.session_state.parent_mol

    # ── Build fragment pool ──────────────────────────────────────────────────
    size_cap = {"Conservative": 4, "Moderate": 8, "Exploratory": 14}[st.session_state.risk]
    active_lib = [
        f for f in core.BUILTIN_LIBRARY
        if st.session_state.categories_on.get(f.category, True) and f.heavy <= size_cap
    ]
    pocket_frags = st.session_state.pocket_frags or []
    if mode == "structure" and pocket_frags and st.session_state.accept_pocket_suggestions:
        chosen = pocket_frags + [f for f in active_lib if f.name not in {g.name for g in pocket_frags}]
    else:
        chosen = active_lib

    # Custom fragments
    for smi in st.session_state.custom_frags_text.strip().splitlines():
        smi = smi.strip()
        ok, _ = core.validate_fragment_smiles(smi)
        if ok:
            chosen.append(core.Frag(f"custom_{len(chosen)}", smi, "custom", core.G()))

    # Valid sites
    selected = st.session_state.selected_atoms
    allow_het = st.session_state.allow_heteroatom_H
    valid_sites = [
        s for s in sorted(selected)
        if mol.GetAtomWithIdx(s).GetTotalNumHs() > 0
        and (allow_het or mol.GetAtomWithIdx(s).GetAtomicNum() == 6)
    ]

    if not valid_sites:
        st.error("No valid attachment sites. Go back to Step 2 and select atoms.")
        if st.button("← Back to atom selection"):
            go(2)
        st.stop()

    site_groups = [tuple(valid_sites)] if (st.session_state.concerted and len(valid_sites) > 1) \
                  else [(s,) for s in valid_sites]

    tot = sum(st.session_state.weights.values()) or 1
    weights = {k: v / tot for k, v in st.session_state.weights.items()}

    avoid_opts = {
        "nitro": st.session_state.avoid_nitro,
        "aldehyde": st.session_state.avoid_aldehyde,
        "reactive_acylhalide": st.session_state.avoid_reactive,
        "azide": st.session_state.avoid_toxic,
        "michael_acceptor": st.session_state.avoid_reactive,
        "epoxide": st.session_state.avoid_toxic,
    }

    # ── Auto-generate on first visit ────────────────────────────────────────
    if st.session_state.analogs_df is None:
        with st.spinner(f"Generating analogs from {len(chosen)} fragments…"):
            df = core.generate_analogs(
                mol,
                selected_atoms=list(selected),
                chosen_frags=chosen,
                site_groups=site_groups,
                weights=weights,
                avoid_opts=avoid_opts,
                max_MW=st.session_state.max_MW,
                max_analogs=min(st.session_state.n_analogs, 20),
                rank_by=st.session_state.get("rank_code", "Balanced (100-pt weights)"),
            )
        st.session_state.analogs_df = df

    df = st.session_state.analogs_df

    if df is None or df.empty:
        st.error("No analogs were generated. Try relaxing your filters in Step 3.")
        if st.button("← Back to settings"):
            go(3)
        st.stop()

    # ── Metrics row ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-box"><div class="val">{len(df)}</div><div class="lbl">Analogs generated</div></div>
      <div class="metric-box"><div class="val">{df.MW.median():.0f}</div><div class="lbl">Median MW (Da)</div></div>
      <div class="metric-box"><div class="val">{df.QED.median():.2f}</div><div class="lbl">Median QED</div></div>
      <div class="metric-box"><div class="val">{df.sim.median():.2f}</div><div class="lbl">Median similarity</div></div>
      <div class="metric-box"><div class="val">{df.fragment_category.value_counts().index[0]}</div><div class="lbl">Top category</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Filter bar ───────────────────────────────────────────────────────────
    with st.expander("🔍 Filter results"):
        fc1, fc2, fc3 = st.columns(3)
        mw_max  = fc1.slider("Max MW", 200.0, 900.0, float(df.MW.max()), 10.0)
        qed_min = fc2.slider("Min QED", 0.0, 1.0, 0.0, 0.05)
        cats_f  = fc3.multiselect("Category", sorted(df.fragment_category.unique()),
                                   default=sorted(df.fragment_category.unique()))

    df_show = df[(df.MW <= mw_max) & (df.QED >= qed_min) & (df.fragment_category.isin(cats_f))]

    # ── Tabs: table / grid ───────────────────────────────────────────────────
    tab_tbl, tab_grid = st.tabs(["📋 Table", "🖼️ Structure grid"])

    with tab_tbl:
        cols_show = ["change", "fragment_category", "MW", "logP", "QED", "ESOL", "SA", "sim", "smiles"]
        st.dataframe(
            df_show[[c for c in cols_show if c in df_show.columns]],
            use_container_width=True, height=380, hide_index=True,
        )

    with tab_grid:
        n_grid = min(len(df_show), 20)
        if n_grid == 0:
            st.info("No analogs match the current filters.")
        else:
            # Build (mol, legend) pairs, dropping any SMILES that fail to parse
            pairs = []
            for i, (smi, chg) in enumerate(
                zip(df_show.smiles.head(n_grid), df_show.change.head(n_grid))
            ):
                m = Chem.MolFromSmiles(str(smi))
                if m is not None:
                    try:
                        AllChem.Compute2DCoords(m)
                        pairs.append((m, f"{i+1}. {chg}"))
                    except Exception:
                        pass

            if not pairs:
                st.info("Could not render any structures from the current results.")
            else:
                mols_g = [p[0] for p in pairs]
                legs = [p[1] for p in pairs]
                try:
                    png = Draw.MolsToGridImage(
                        mols_g, legends=legs, molsPerRow=4,
                        subImgSize=(280, 210),
                        returnPNG=True,
                    )
                    st.image(png, use_container_width=True)
                except Exception as e:
                    st.warning(f"Structure grid could not be rendered ({e}). "
                               "See the Table tab for full results.")

    # ── Navigation ───────────────────────────────────────────────────────────
    st.write("")
    col_back, col_regen, col_next = st.columns([1, 2, 3])
    with col_back:
        if st.button("← Back"):
            go(3)
    with col_regen:
        if st.button("↺ Regenerate"):
            st.session_state.analogs_df = None
            st.rerun()
    with col_next:
        next_label = "Docking & cIFP →" if mode == "structure" else "Docking →"
        if st.button(next_label, type="primary"):
            go(5)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Docking & cIFP  (structure track only)
# ─────────────────────────────────────────────────────────────────────────────

elif step == 5:
    df_analogs = st.session_state.analogs_df
    if df_analogs is None or df_analogs.empty:
        st.warning("No analogs yet. Go back to Step 4.")
        st.stop()

    acd_ok    = bool(shutil.which("acd"))
    obabel_ok = bool(shutil.which("obabel"))
    work      = get_work_dir()
    dock_in   = work / "docking_inputs"
    dock_in.mkdir(parents=True, exist_ok=True)

    info_card("Docking predicts how tightly each designed analog binds to the target protein. "
              "Requires <strong>Anyone Can Dock</strong> (acd) and <strong>OpenBabel</strong>.")

    # ── Status badges ────────────────────────────────────────────────────────
    bcol1, bcol2 = st.columns(2)
    bcol1.markdown(
        f'<div style="padding:0.6rem 1rem;border-radius:8px;'
        f'background:{"#E6F4EA" if acd_ok else "#FDECEA"};'
        f'color:{"#1E7E34" if acd_ok else "#B00020"};font-size:0.85rem;">'
        f'{"✅ acd available" if acd_ok else "❌ acd not found — pip install anyonecandock"}</div>',
        unsafe_allow_html=True
    )
    bcol2.markdown(
        f'<div style="padding:0.6rem 1rem;border-radius:8px;'
        f'background:{"#E6F4EA" if obabel_ok else "#FDECEA"};'
        f'color:{"#1E7E34" if obabel_ok else "#B00020"};font-size:0.85rem;">'
        f'{"✅ obabel available" if obabel_ok else "❌ obabel not found — apt install openbabel"}</div>',
        unsafe_allow_html=True
    )

    # ── Receptor: ligand track must load one here ────────────────────────────
    if mode == "ligand" and not st.session_state.receptor_path:
        st.divider()
        st.markdown("### Choose a target protein")
        info_card("You designed analogs without a structure. To dock them, pick the protein they bind to.")
        rec_src = st.radio(
            "Load receptor from",
            ["🔍 Search RCSB", "#️⃣ PDB ID", "📁 Upload file"],
            horizontal=True, key="dock_rec_src",
        )
        if rec_src == "🔍 Search RCSB":
            dq = st.text_input("Search RCSB PDB", placeholder="e.g. EGFR kinase, JAK2", key="dock_rcsb_q")
            if st.button("Search", key="dock_rcsb_btn") and dq.strip():
                with st.spinner("Searching RCSB PDB…"):
                    st.session_state["_dock_rcsb_results"] = core.search_rcsb(dq.strip(), max_results=6)
            for r in st.session_state.get("_dock_rcsb_results", []):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.caption(f"**{r['id']}** — {r['title']}  ({r['resolution']} · {r['organism']})")
                with c2:
                    if st.button("Use", key=f"dock_rcsb_{r['id']}"):
                        with st.spinner(f"Downloading {r['id']}…"):
                            try:
                                path = core.download_pdb(r["id"], work)
                                prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                                st.session_state.receptor_path   = path
                                st.session_state.protein_path    = prot
                                st.session_state.complex_path    = path
                                st.session_state.ref_ligand_path = lig
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not download: {e}")

        elif rec_src == "#️⃣ PDB ID":
            pdb_id = st.text_input("4-letter PDB ID", value="", max_chars=4, placeholder="e.g. 1M17", key="dock_pdb_id")
            hint("Example: 1M17 is EGFR.")
            if st.button("Load receptor", key="dock_load_rec") and pdb_id.strip():
                with st.spinner("Downloading from RCSB…"):
                    try:
                        path = core.download_pdb(pdb_id.strip().upper(), work)
                        prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                        st.session_state.receptor_path   = path
                        st.session_state.protein_path    = prot
                        st.session_state.complex_path    = path
                        st.session_state.ref_ligand_path = lig
                        st.success(f"Receptor loaded ({pdb_id.upper()}) ✅")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not download: {e}")
        else:
            up = st.file_uploader("Upload .pdb or .cif file", type=["pdb", "cif"], key="dock_upload")
            if up:
                raw = work / up.name
                raw.write_bytes(up.read())
                try:
                    path = core.cif_to_pdb_if_needed(str(raw))
                    prot, lig, _ = core.split_protein_ligand(path, work_dir=work / "receptor")
                    st.session_state.receptor_path   = path
                    st.session_state.protein_path    = prot
                    st.session_state.complex_path    = path
                    st.session_state.ref_ligand_path = lig
                    st.success("Receptor uploaded ✅")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not process file: {e}")

    receptor = st.session_state.receptor_path
    if receptor:
        st.success(f"Target receptor: `{Path(receptor).name}`")

    # ── Ligand list (parent + all analogs) ───────────────────────────────────
    st.divider()
    rows = [{"compound": "original_ligand", "smiles": st.session_state.parent_smiles}]
    for i, r in df_analogs.iterrows():
        rows.append({"compound": f"{st.session_state.parent_name}_A{i+1}", "smiles": r.smiles})
    lig_df = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
    st.session_state.docking_ligands = lig_df

    smi_path = dock_in / "compounds.smi"
    with open(smi_path, "w") as fh:
        for _, r in lig_df.iterrows():
            fh.write(f"{r.smiles}\t{r.compound}\n")

    with st.expander(f"Ligand list — {len(lig_df)} compounds (parent + analogs)"):
        st.dataframe(lig_df, use_container_width=True, hide_index=True, height=200)

    # ── Docking settings ─────────────────────────────────────────────────────
    st.markdown("### Docking settings")
    d1, d2 = st.columns(2)
    with d1:
        exhaustiveness = st.slider("Exhaustiveness", 1, 32, 8)
        hint("Higher = more thorough but slower. 8 is standard.")
        num_poses = st.slider("Poses per compound", 1, 20, 10)
    with d2:
        dock_ph = st.number_input("pH", value=7.4, step=0.1)
        hint("Used for protonation state assignment.")
        dock_out_dir = str(work / "docking_out")

    with st.expander("⚙️ Advanced docking options"):
        bx = st.number_input("Box X (Å)", value=16.0)
        by = st.number_input("Box Y (Å)", value=16.0)
        bz = st.number_input("Box Z (Å)", value=16.0)

    # ── Run / fallback ───────────────────────────────────────────────────────
    if acd_ok and obabel_ok and receptor:
        if st.button("Run docking", type="primary"):
            n = len(lig_df)
            progress = st.progress(0.0, text=f"Preparing to dock {n} compounds…")
            status = st.empty()
            live_table = st.empty()
            dock_results = []
            any_fail = False
            full_log = []

            # Load reference ligand for RMSD (if available)
            ref_mol = None
            ref_pdb = st.session_state.ref_ligand_path
            if ref_pdb and os.path.exists(str(ref_pdb)):
                try:
                    ref_mol = Chem.MolFromPDBFile(str(ref_pdb), removeHs=True)
                except Exception:
                    pass

            for i, row in lig_df.iterrows():
                compound = str(row["compound"])
                smi = str(row["smiles"])
                frac = i / n
                progress.progress(frac, text=f"Docking {i+1} of {n}:  {compound}")
                status.markdown(
                    f'<div style="font-size:0.85rem;color:#8B7355;">'
                    f'⏳ Running AutoDock Vina on <strong>{compound}</strong>…</div>',
                    unsafe_allow_html=True,
                )

                cmd = core.build_acd_dock_cmd(
                    receptor=receptor,
                    smiles=smi,
                    name=compound,
                    ph=dock_ph,
                    output_dir=dock_out_dir,
                    save_poses=True,
                )
                rc, output = core.run_command(cmd)
                full_log.append(f"=== {compound} (exit {rc}) ===\n{output}\n")
                if rc != 0:
                    any_fail = True

                # Analyse poses: BE + RMSD for top pose and min-RMSD pose
                summary = core.summarize_docking_for_compound(
                    out_dir=dock_out_dir,
                    compound=compound,
                    smiles=smi,
                    ref_mol=ref_mol,
                    ref_pdb_path=ref_pdb,
                )
                summary["dock_status"] = "✅" if rc == 0 else "❌"
                dock_results.append(summary)

                # Live preview table (lightweight columns)
                preview_df = pd.DataFrame([{
                    "compound": r["compound"],
                    "status": r["dock_status"],
                    "top_BE": r.get("top_BE", "—"),
                    "top_RMSD": r.get("top_RMSD", "—"),
                } for r in dock_results])
                live_table.dataframe(preview_df, use_container_width=True, hide_index=True)

            progress.progress(1.0, text=f"Finished docking {n} compounds")
            status.empty()
            (work / "acd_batch.log").write_text("\n".join(full_log))

            if any_fail:
                st.warning("Docking finished, but some compounds failed. See the log below.")
            else:
                st.success(f"Docking finished — all {n} compounds ✅")

            # ── Full results table ───────────────────────────────────────────
            st.divider()
            st.markdown("### Docking results")

            if ref_mol:
                st.info("RMSD computed against the co-crystal ligand pose.")
            else:
                st.caption("No reference ligand available — RMSD columns will be empty.")

            # Build full results DataFrame
            result_rows = []
            for r in dock_results:
                result_rows.append({
                    "compound": r["compound"],
                    "SMILES": r["smiles"],
                    "status": r["dock_status"],
                    "n_poses": r.get("n_poses", 0),
                    "top_BE (kcal/mol)": r.get("top_BE"),
                    "top_RMSD (Å)": r.get("top_RMSD"),
                    "minRMSD_BE (kcal/mol)": r.get("minRMSD_BE"),
                    "minRMSD (Å)": r.get("minRMSD_RMSD"),
                })
            results_df = pd.DataFrame(result_rows)

            # Sort by top_BE
            if "top_BE (kcal/mol)" in results_df.columns:
                results_df = results_df.sort_values("top_BE (kcal/mol)", na_position="last").reset_index(drop=True)

            st.dataframe(results_df, use_container_width=True, hide_index=True)
            st.session_state.docking_summary = results_df

            # ── 2D structure gallery ─────────────────────────────────────────
            st.markdown("### 2D Structures")
            grid_mols, grid_legs = [], []
            for r in result_rows:
                m = Chem.MolFromSmiles(str(r["SMILES"]))
                if m:
                    try:
                        AllChem.Compute2DCoords(m)
                        be_str = f"BE={r['top_BE (kcal/mol)']}" if r.get("top_BE (kcal/mol)") else ""
                        rmsd_str = f"RMSD={r['top_RMSD (Å)']}" if r.get("top_RMSD (Å)") else ""
                        grid_mols.append(m)
                        grid_legs.append(f"{r['compound']}\n{be_str}  {rmsd_str}")
                    except Exception:
                        pass
            if grid_mols:
                try:
                    png = Draw.MolsToGridImage(
                        grid_mols, legends=grid_legs, molsPerRow=4,
                        subImgSize=(280, 210), returnPNG=True,
                    )
                    st.image(png, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not render structure grid: {e}")

            # ── CSV download ─────────────────────────────────────────────────
            st.divider()
            st.markdown("### Download docking results")

            # Build CSV with all columns
            csv_rows = []
            for r in result_rows:
                csv_rows.append({
                    "compound": r["compound"],
                    "SMILES": r["SMILES"],
                    "status": r["status"],
                    "n_poses": r["n_poses"],
                    "top_pose_BE_kcal_mol": r["top_BE (kcal/mol)"],
                    "top_pose_RMSD_vs_crystal_A": r["top_RMSD (Å)"],
                    "min_RMSD_pose_BE_kcal_mol": r["minRMSD_BE (kcal/mol)"],
                    "min_RMSD_vs_crystal_A": r["minRMSD (Å)"],
                })
            csv_df = pd.DataFrame(csv_rows)
            csv_bytes = csv_df.to_csv(index=False).encode()

            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Docking results CSV",
                    data=csv_bytes,
                    file_name=f"{st.session_state.parent_name}_docking_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                # Combined SMI file
                smi_out = "\n".join(f"{r['SMILES']}\t{r['compound']}" for r in result_rows)
                st.download_button(
                    "⬇️ Docked compounds SMILES",
                    data=smi_out.encode(),
                    file_name=f"{st.session_state.parent_name}_docked.smi",
                    mime="text/plain",
                    use_container_width=True,
                )

            with st.expander("ACD log"):
                st.text("\n".join(full_log)[-4000:])
    elif not receptor:
        st.info("Load a target receptor above to enable docking.")
    else:
        st.info("ACD or OpenBabel is not installed here. Download the ligand file and dock externally.")
        st.download_button("⬇️ Download compounds.smi", data=smi_path.read_text(),
                           file_name="compounds.smi", mime="text/plain")

    # ── cIFP (structure track only — needs split protein) ────────────────────
    if mode == "structure" and st.session_state.protein_path:
        st.divider()
        st.markdown("### Interaction fingerprints (cIFP)")
        hint("After docking, compute which residues each pose contacts.")
        if st.button("Run cIFP analysis"):
            plip_available = bool(shutil.which("plipcmd") or shutil.which("plip"))
            cifp_dir = work / "plip_cifp" / "complexes"
            cifp_dir.mkdir(parents=True, exist_ok=True)
            pose_dir = work / "docking_out" / "selected_pose_pdbs"
            pose_pdbs = list(pose_dir.glob("*.pdb")) if pose_dir.exists() else []
            if not pose_pdbs:
                st.warning("No docked poses found yet. Run docking first.")
            else:
                rows_c = []
                for p in pose_pdbs[:30]:
                    cpx = str(cifp_dir / f"{p.stem}_complex.pdb")
                    core.combine_protein_ligand_pdb(st.session_state.protein_path, str(p), cpx)
                    feats, method = [], "distance"
                    if plip_available:
                        xml, _ = core.run_plip(cpx, str(work / "plip_cifp" / "plip_out"), p.stem)
                        feats = core.parse_plip_xml(xml) if xml else []
                        method = "PLIP" if feats else "distance"
                    if not feats:
                        feats = core.distance_contact_cifp(cpx, cutoff=4.0)
                    rows_c.append({"compound": p.stem, "method": method,
                                   "n_interactions": len(feats), "features": ";".join(feats)})
                cifp_df = pd.DataFrame(rows_c)
                st.session_state.cifp_results = cifp_df
                st.success(f"cIFP computed for {len(cifp_df)} poses")
                st.dataframe(cifp_df, use_container_width=True, hide_index=True)

    # ── Navigation ───────────────────────────────────────────────────────────
    st.write("")
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            go(4)
    with col_next:
        if st.button("Export results →", type="primary"):
            go(len(steps))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 (ligand) / STEP 6 (structure) – Export
# ─────────────────────────────────────────────────────────────────────────────

elif step == len(steps):
    df = st.session_state.analogs_df
    if df is None or df.empty:
        st.warning("No analogs generated yet. Complete the earlier steps first.")
        st.stop()

    parent_name = st.session_state.parent_name or "compound"
    work = get_work_dir()

    info_card("Your analogs are ready. Download the table, SMILES file, 3D structures, or a full ZIP archive.")

    # ── Quick downloads ──────────────────────────────────────────────────────
    st.markdown("### Download analogs")
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ Analog table (CSV)",
            data=df.to_csv(index=False).encode(),
            file_name=f"{parent_name}_analogs.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        smi_lines = "\n".join(f"{r.smiles}\t{parent_name}_A{i+1}" for i, r in df.iterrows())
        st.download_button(
            "⬇️ SMILES file (.smi)",
            data=smi_lines.encode(),
            file_name=f"{parent_name}_analogs.smi",
            mime="text/plain",
            use_container_width=True,
        )

    # ── 3D SDF ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Generate 3D structures")
    hint("Creates a 3D conformer for each analog — useful for visualisation or further docking.")

    g1, g2 = st.columns(2)
    with g1:
        fmt_sel = st.multiselect("Output formats", ["SDF", "PDB", "MOL2"], default=["SDF"])
    with g2:
        mmff_opt = st.checkbox("MMFF geometry optimisation", value=True)

    if st.button("Generate 3D structures", type="primary"):
        lig_table = df[["smiles"]].copy()
        lig_table["compound"] = [f"{parent_name}_A{i+1}" for i in range(len(df))]
        out_dir = work / "ligands_3d"
        with st.spinner("Building 3D conformers…"):
            manifest = core.generate_3d_ligand_files(lig_table, out_dir, formats=fmt_sel, mmff=mmff_opt)
        ok_count = int((manifest.status == "ok").sum())
        st.success(f"3D files generated: {ok_count} / {len(manifest)}")
        st.dataframe(manifest, use_container_width=True, hide_index=True)
        combined = out_dir / "all_ligands_3d.sdf"
        if combined.exists():
            st.download_button(
                "⬇️ Download combined SDF",
                data=combined.read_bytes(),
                file_name=f"{parent_name}_3d.sdf",
                mime="chemical/x-mdl-sdfile",
                use_container_width=True,
            )

    # ── Full ZIP ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Full archive")
    hint("Everything in one ZIP — analog table, SMILES, 3D files, docking results, and a session summary.")

    if st.button("Build ZIP archive", use_container_width=True):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"{parent_name}_analogs.csv", df.to_csv(index=False))
            z.writestr(f"{parent_name}_analogs.smi", smi_lines)
            for subdir in ["ligands_3d", "docking_out"]:
                p = work / subdir
                if p.exists():
                    for f in p.rglob("*.*"):
                        if f.is_file():
                            z.write(f, arcname=f"{subdir}/{f.relative_to(p)}")
            cifp = st.session_state.cifp_results
            if cifp is not None and not cifp.empty:
                z.writestr("cifp_results.csv", cifp.to_csv(index=False))
            z.writestr("session_info.json", json.dumps({
                "parent_smiles":    st.session_state.parent_smiles,
                "parent_name":      parent_name,
                "mode":             mode,
                "n_analogs":        len(df),
                "selected_atoms":   sorted(st.session_state.selected_atoms),
                "risk":             st.session_state.risk,
                "rank_by":          st.session_state.rank_by,
            }, indent=2))
        buf.seek(0)
        st.download_button(
            "⬇️ Download full ZIP",
            data=buf.getvalue(),
            file_name=f"{parent_name}_analog_builder_results.zip",
            mime="application/zip",
            use_container_width=True,
        )

    # ── Session summary ──────────────────────────────────────────────────────
    st.divider()
    with st.expander("Session summary"):
        st.json({
            "mode":            mode,
            "parent_smiles":   st.session_state.parent_smiles,
            "parent_name":     parent_name,
            "selected_atoms":  sorted(st.session_state.selected_atoms),
            "analogs_generated": len(df),
            "top_category":    df.fragment_category.value_counts().index[0] if len(df) else "—",
            "receptor_loaded": bool(st.session_state.receptor_path),
            "docking_run":     st.session_state.docking_ligands is not None,
        })

    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Back"):
            go(step - 1)
