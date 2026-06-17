"""
app.py — Streamlit web application for the Analog Designer.

Workflow mirrors the original notebook:
  Step 1  · Parent ligand + design route
  Step 2  · Atom selection
  Step 3  · Design settings (goals, fragment options, pocket guidance)
  Step 4  · Generate analogs
  Step 5  · Docking + PLIP/cIFP validation (optional)
  Export  · Download results
"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from collections import Counter

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

import core

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Analog Designer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

DEFAULTS = {
    "parent_smiles": "O=C1OC2=C(C=CC=C2)C(O)=C1C(C3=CC=CC=C3)C4=C(O)C5=CC=CC=C5OC4=O",
    "parent_name": "parent",
    "parent_mol": None,
    "design_workflow": "Ligand-based",
    # Step 2
    "selected_atoms": set(),
    "concerted": False,
    # Step 3
    "weights": {"potency": 30, "selectivity": 10, "solubility": 25,
                "metabolic": 15, "synthesis": 10, "novelty": 10},
    "categories_on": {k: True for k in core.CATEGORY_BASE_GOALS},
    "risk": "Moderate",
    "n_analogs": 50,
    "max_MW": 600.0,
    "avoid_nitro": True,
    "avoid_aldehyde": True,
    "avoid_reactive": True,
    "avoid_toxic": True,
    "avoid_large_MW": True,
    "rank_by": "Balanced (100-pt weights)",
    "custom_frags_text": "",
    "allow_heteroatom_H": False,
    "pocket_guidance_source": "Off",
    "pocket_residue_text": "",
    "accept_pocket_suggestions": True,
    "pocket_mode": "Blend with selected options",
    "max_pocket_frags": 6,
    # Step 4 results
    "analogs_df": None,
    "pocket_frags": [],
    "strategy_df": None,
    # Step 5 docking
    "receptor_pdb_id": "1M17",
    "receptor_path": None,
    "protein_path": None,
    "complex_path": None,
    "ref_ligand_path": None,
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
        st.session_state.work_dir = Path(tempfile.mkdtemp(prefix="analog_designer_"))
    return Path(st.session_state.work_dir)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🧬 Analog Designer")
st.sidebar.markdown("**Structure-guided analog generation**")

PAGES = [
    "1 · Parent Ligand",
    "2 · Atom Selection",
    "3 · Design Settings",
    "4 · Generate Analogs",
    "5 · Docking & cIFP",
    "📦 Export",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.caption(
    "Powered by RDKit · Anyone Can Dock · pKaNET · PLIP\n\n"
    "Fragment library: " + str(len(core.BUILTIN_LIBRARY)) + " built-in fragments"
)


# ---------------------------------------------------------------------------
# Helper: molecule SVG
# ---------------------------------------------------------------------------

def mol_svg(mol, highlight=None, size=(480, 380)):
    if mol is None:
        return ""
    return core.draw_mol_svg(mol, highlight=list(highlight or []), size=size)


def svg_to_img_tag(svg: str) -> str:
    import base64
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" style="max-width:100%;">'


# ---------------------------------------------------------------------------
# PAGE 1 · Parent Ligand
# ---------------------------------------------------------------------------

if page == "1 · Parent Ligand":
    st.title("Step 1 · Parent Ligand & Design Route")

    col1, col2 = st.columns([2, 1])
    with col1:
        smiles_input = st.text_area(
            "Parent SMILES",
            value=st.session_state.parent_smiles,
            height=80,
            help="Paste the SMILES of the parent compound you want to elaborate.",
        )
        parent_name = st.text_input("Compound name", value=st.session_state.parent_name)
        workflow = st.radio(
            "Design workflow",
            ["Ligand-based", "Structure-based"],
            index=0 if st.session_state.design_workflow == "Ligand-based" else 1,
            horizontal=True,
            help=(
                "**Ligand-based**: skip receptor docking; use the library and pocket residues you paste manually.\n\n"
                "**Structure-based**: dock the parent into a receptor first, then use the pocket environment automatically."
            ),
        )

        if st.button("Load parent", type="primary"):
            mol = Chem.MolFromSmiles(smiles_input.strip())
            if mol is None:
                st.error("Invalid SMILES. Please check and try again.")
            else:
                AllChem.Compute2DCoords(mol)
                st.session_state.parent_smiles = smiles_input.strip()
                st.session_state.parent_name = parent_name.strip() or "parent"
                st.session_state.parent_mol = mol
                st.session_state.design_workflow = workflow
                st.session_state.selected_atoms = set()
                st.session_state.analogs_df = None
                st.success("Parent loaded ✅")

    with col2:
        mol = st.session_state.parent_mol
        if mol:
            attachable = core.attachable_atom_indices(mol, carbon_only=False)
            c_only = core.attachable_atom_indices(mol, carbon_only=True)
            st.markdown("**Molecule preview**")
            svg = mol_svg(mol, highlight=c_only)
            st.markdown(svg_to_img_tag(svg), unsafe_allow_html=True)
            st.caption(
                f"Atoms: {mol.GetNumAtoms()} | "
                f"Recommended C–H sites: {c_only} | "
                f"All H-bearing: {attachable}"
            )
        else:
            st.info("Load a valid SMILES to see the molecule.")

    # Structure-based: receptor download
    if st.session_state.design_workflow == "Structure-based" and st.session_state.parent_mol:
        st.divider()
        st.subheader("1B · Receptor / Complex Setup")
        st.info(
            "For structure-based design, provide a receptor PDB. "
            "The parent ligand will be docked and the pocket environment analyzed automatically."
        )
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            rec_src = st.radio("Receptor source", ["PDB ID", "Upload PDB file"], horizontal=True)
            if rec_src == "PDB ID":
                pdb_id = st.text_input("PDB ID", value=st.session_state.receptor_pdb_id, max_chars=4)
                if st.button("Download receptor"):
                    work = get_work_dir()
                    with st.spinner("Downloading from RCSB..."):
                        try:
                            path = core.download_pdb(pdb_id.strip().upper(), work)
                            protein_path, ref_lig, cands = core.split_protein_ligand(
                                path, work_dir=work / "receptor"
                            )
                            st.session_state.receptor_pdb_id = pdb_id.strip().upper()
                            st.session_state.receptor_path = path
                            st.session_state.protein_path = protein_path
                            st.session_state.ref_ligand_path = ref_lig
                            st.session_state.complex_path = path
                            st.success(f"Receptor downloaded: {path}")
                            if ref_lig:
                                st.info(f"Reference ligand detected: {ref_lig}")
                        except Exception as e:
                            st.error(f"Download failed: {e}")
            else:
                uploaded = st.file_uploader("Upload receptor PDB", type=["pdb", "cif"])
                if uploaded and st.button("Process uploaded receptor"):
                    work = get_work_dir()
                    raw_path = work / uploaded.name
                    raw_path.write_bytes(uploaded.read())
                    try:
                        pdb_path = core.cif_to_pdb_if_needed(str(raw_path))
                        protein_path, ref_lig, cands = core.split_protein_ligand(
                            pdb_path, work_dir=work / "receptor"
                        )
                        st.session_state.receptor_path = pdb_path
                        st.session_state.protein_path = protein_path
                        st.session_state.ref_ligand_path = ref_lig
                        st.session_state.complex_path = pdb_path
                        st.success("Receptor processed ✅")
                    except Exception as e:
                        st.error(f"Processing failed: {e}")

        with rec_col2:
            if st.session_state.receptor_path:
                st.success(f"Receptor ready: `{st.session_state.receptor_path}`")
                if st.session_state.ref_ligand_path:
                    st.success(f"Reference ligand: `{st.session_state.ref_ligand_path}`")


# ---------------------------------------------------------------------------
# PAGE 2 · Atom Selection
# ---------------------------------------------------------------------------

elif page == "2 · Atom Selection":
    st.title("Step 2 · Select Attachment Atom(s)")
    mol = st.session_state.parent_mol

    if mol is None:
        st.warning("Go back to Step 1 and load a parent SMILES first.")
    else:
        attachable = core.attachable_atom_indices(mol, carbon_only=False)
        c_only = core.attachable_atom_indices(mol, carbon_only=True)

        st.markdown(
            "Select one or more atoms where a new substituent will be attached. "
            "Tick atoms in the list below and they will be highlighted in the structure."
        )

        col_mol, col_sel = st.columns([1, 1])

        with col_sel:
            st.markdown("**Recommended C–H sites**")
            new_selected = set()
            for idx in attachable:
                atom = mol.GetAtomWithIdx(idx)
                label = (
                    f"Atom {idx} ({atom.GetSymbol()}) "
                    f"[{('C–H' if atom.GetAtomicNum()==6 else 'N/O/S–H')}]"
                )
                default = idx in st.session_state.selected_atoms
                if st.checkbox(label, value=default, key=f"atom_{idx}"):
                    new_selected.add(idx)

            st.session_state.selected_atoms = new_selected

            st.divider()
            concerted = st.checkbox(
                "Concerted multi-position mode",
                value=st.session_state.concerted,
                help=(
                    "If ON: one analog modifies **all** selected atoms together with the same fragment.\n\n"
                    "If OFF (default): one analog modifies **one** selected atom at a time."
                ),
            )
            st.session_state.concerted = concerted

            allow_het = st.checkbox(
                "Allow heteroatom–H substitution (N–H, O–H, S–H)",
                value=st.session_state.allow_heteroatom_H,
            )
            st.session_state.allow_heteroatom_H = allow_het

        with col_mol:
            svg = mol_svg(mol, highlight=sorted(st.session_state.selected_atoms))
            st.markdown(svg_to_img_tag(svg), unsafe_allow_html=True)

        if st.session_state.selected_atoms:
            st.success(f"Selected atoms: {sorted(st.session_state.selected_atoms)}")
        else:
            st.info("No atoms selected yet. Tick at least one atom above.")


# ---------------------------------------------------------------------------
# PAGE 3 · Design Settings
# ---------------------------------------------------------------------------

elif page == "3 · Design Settings":
    st.title("Step 3 · Design Settings")

    tabs = st.tabs(["3A Goals", "3B Fragment Options", "3C Pocket Guidance"])

    # ---- 3A Goal weights ----
    with tabs[0]:
        st.subheader("Goal weights (auto-normalised)")
        w = st.session_state.weights
        cols = st.columns(3)
        keys = list(w.keys())
        new_w = {}
        for i, k in enumerate(keys):
            with cols[i % 3]:
                new_w[k] = st.slider(k.capitalize(), 0, 100, int(w[k]), step=5)
        st.session_state.weights = new_w
        tot = sum(new_w.values()) or 1
        norm = {k: round(v / tot, 3) for k, v in new_w.items()}
        st.markdown("**Normalised weights:** " + " | ".join(f"{k}: {v:.2f}" for k, v in norm.items()))

    # ---- 3B Fragment options ----
    with tabs[1]:
        st.subheader("Fragment categories and filters")
        cat_col1, cat_col2 = st.columns(2)
        cats = list(core.CATEGORY_BASE_GOALS.keys())
        new_cats = {}
        for i, cat in enumerate(cats):
            col = cat_col1 if i < len(cats) // 2 else cat_col2
            with col:
                new_cats[cat] = st.checkbox(cat.replace("_", " ").capitalize(),
                                             value=st.session_state.categories_on.get(cat, True),
                                             key=f"cat_{cat}")
        st.session_state.categories_on = new_cats

        st.divider()
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            risk = st.selectbox("Risk / size cap", ["Conservative", "Moderate", "Exploratory", "Scaffold hopping"],
                                index=["Conservative", "Moderate", "Exploratory", "Scaffold hopping"].index(
                                    st.session_state.risk))
            st.session_state.risk = risk

            n_analogs = st.slider("Target analog count", 5, 500, st.session_state.n_analogs, step=5)
            st.session_state.n_analogs = n_analogs

            max_MW = st.number_input("Max MW (Da)", value=st.session_state.max_MW, step=25.0)
            st.session_state.max_MW = max_MW

        with filter_col2:
            st.markdown("**Structural filters**")
            st.session_state.avoid_nitro = st.checkbox("Avoid nitro groups", value=st.session_state.avoid_nitro)
            st.session_state.avoid_aldehyde = st.checkbox("Avoid aldehydes", value=st.session_state.avoid_aldehyde)
            st.session_state.avoid_reactive = st.checkbox("Avoid reactive acyl halides / Michael acceptors",
                                                           value=st.session_state.avoid_reactive)
            st.session_state.avoid_toxic = st.checkbox("Avoid toxic flags (azides, epoxides)",
                                                        value=st.session_state.avoid_toxic)
            st.session_state.avoid_large_MW = st.checkbox("Enforce MW cap", value=st.session_state.avoid_large_MW)

        st.divider()
        st.session_state.rank_by = st.selectbox(
            "Rank analogs by",
            ["Balanced (100-pt weights)", "Similarity to parent", "Solubility (ESOL)",
             "ADMET (QED)", "Synthetic feasibility", "Binding proxy (heuristic)"],
            index=["Balanced (100-pt weights)", "Similarity to parent", "Solubility (ESOL)",
                   "ADMET (QED)", "Synthetic feasibility", "Binding proxy (heuristic)"].index(
                st.session_state.rank_by),
        )

        st.divider()
        st.markdown("**Custom fragments** (one SMILES with `[*]` per line)")
        st.session_state.custom_frags_text = st.text_area(
            "Custom fragment SMILES",
            value=st.session_state.custom_frags_text,
            height=100,
            label_visibility="collapsed",
        )

    # ---- 3C Pocket guidance ----
    with tabs[2]:
        st.subheader("Pocket-guided functional group suggestions")
        source = st.radio(
            "Pocket guidance source",
            ["Off", "Manual pasted residues", "Auto from uploaded complex"],
            index=["Off", "Manual pasted residues", "Auto from uploaded complex"].index(
                st.session_state.pocket_guidance_source
            ),
        )
        st.session_state.pocket_guidance_source = source

        if source == "Manual pasted residues":
            st.session_state.pocket_residue_text = st.text_area(
                "Pocket residues (e.g. ASP315, LYS89, TYR102 or D315 K89 Y102)",
                value=st.session_state.pocket_residue_text,
                height=80,
            )

        elif source == "Auto from uploaded complex":
            if st.session_state.complex_path and os.path.exists(str(st.session_state.complex_path)):
                st.success(f"Using complex: `{st.session_state.complex_path}`")
                if st.button("Run pocket analysis"):
                    with st.spinner("Analyzing pocket environment..."):
                        try:
                            pocket_df, contact_df, growth_df, lig_atoms = core.analyze_complex_distance_shell(
                                st.session_state.complex_path
                            )
                            residue_codes = [x for x in growth_df["aa_one"].tolist() if x]
                            st.session_state.pocket_residue_text = " ".join(
                                [core.AA_ONE_TO_THREE.get(r, r) for r in residue_codes]
                            )
                            st.success(
                                f"Pocket: {len(pocket_df)} residues | "
                                f"Contacted: {len(contact_df)} | "
                                f"Growth opportunities: {len(growth_df)}"
                            )
                            st.dataframe(growth_df[["residue_label", "min_dist_to_ligand_A", "property_tags"]].head(20))
                        except Exception as e:
                            st.error(f"Pocket analysis failed: {e}")
            else:
                st.warning("No complex file found. Go to Step 1B and load/download a receptor first.")

        if source != "Off":
            st.divider()
            pc1, pc2 = st.columns(2)
            with pc1:
                st.session_state.accept_pocket_suggestions = st.checkbox(
                    "Accept pocket-guided suggestions in Step 4",
                    value=st.session_state.accept_pocket_suggestions,
                )
                st.session_state.max_pocket_frags = st.slider(
                    "Max suggested fragments", 3, 20, st.session_state.max_pocket_frags
                )
            with pc2:
                st.session_state.pocket_mode = st.radio(
                    "Pocket guidance mode",
                    ["Blend with selected options", "Use pocket-guided groups only"],
                    index=0 if st.session_state.pocket_mode == "Blend with selected options" else 1,
                )

            residue_codes = core.parse_pocket_residues(st.session_state.pocket_residue_text)
            if residue_codes:
                active_lib = [f for f in core.BUILTIN_LIBRARY if st.session_state.categories_on.get(f.category, True)]
                strategy_df, ratio_df, pocket_frags = core.suggest_fragments_from_residues(
                    residue_codes, active_lib, st.session_state.max_pocket_frags
                )
                st.session_state.pocket_frags = pocket_frags
                st.session_state.strategy_df = strategy_df

                st.markdown("**Pocket property profile**")
                if not ratio_df.empty:
                    st.dataframe(ratio_df, use_container_width=True)
                if not strategy_df.empty:
                    st.markdown("**Suggested interaction strategies**")
                    st.dataframe(strategy_df, use_container_width=True)
                st.markdown("**Suggested fragments**")
                frag_df = pd.DataFrame([
                    {"name": f.name, "smiles": f.smiles, "category": f.category}
                    for f in pocket_frags
                ])
                if not frag_df.empty:
                    st.dataframe(frag_df, use_container_width=True)
            else:
                if source != "Off":
                    st.info("No residues parsed yet. Paste residue names or run pocket analysis above.")


# ---------------------------------------------------------------------------
# PAGE 4 · Generate Analogs
# ---------------------------------------------------------------------------

elif page == "4 · Generate Analogs":
    st.title("Step 4 · Generate Analogs")

    mol = st.session_state.parent_mol
    if mol is None:
        st.warning("Return to Step 1 and load a parent SMILES.")
        st.stop()

    selected = st.session_state.selected_atoms
    if not selected:
        st.warning("Return to Step 2 and select at least one attachment atom.")
        st.stop()

    # Determine valid sites
    allow_het = st.session_state.allow_heteroatom_H
    valid_sites = [
        s for s in sorted(selected)
        if mol.GetAtomWithIdx(s).GetTotalNumHs() > 0
        and (allow_het or mol.GetAtomWithIdx(s).GetAtomicNum() == 6)
    ]

    if not valid_sites:
        st.error(
            "None of the selected atoms have replaceable H. "
            "Enable heteroatom–H substitution in Step 2, or choose different atoms."
        )
        st.stop()

    # Build site groups
    concerted = st.session_state.concerted
    site_groups = [tuple(valid_sites)] if (concerted and len(valid_sites) > 1) else [(s,) for s in valid_sites]

    # Build fragment pool
    size_cap = {"Conservative": 4, "Moderate": 8, "Exploratory": 14, "Scaffold hopping": 20}[
        st.session_state.risk
    ]
    active_lib = [
        f for f in core.BUILTIN_LIBRARY
        if st.session_state.categories_on.get(f.category, True) and f.heavy <= size_cap
    ]

    pocket_frags = st.session_state.pocket_frags or []
    pocket_names = {f.name for f in pocket_frags}
    pocket_accepted = (
        st.session_state.accept_pocket_suggestions
        and bool(pocket_frags)
        and st.session_state.pocket_guidance_source != "Off"
    )

    # Custom fragments
    custom_smis = [
        s.strip()
        for s in st.session_state.custom_frags_text.strip().splitlines()
        if s.strip()
    ]
    custom_frags = []
    for smi in custom_smis:
        ok, _ = core.validate_fragment_smiles(smi)
        if ok:
            custom_frags.append(core.Frag(f"custom_{len(custom_frags)+1}", smi, "custom", core.G()))

    # Choose final fragment pool
    if pocket_accepted and st.session_state.pocket_mode == "Use pocket-guided groups only":
        chosen = pocket_frags[:st.session_state.n_analogs]
    elif pocket_accepted:
        combined = list(active_lib) + [f for f in pocket_frags if f.name not in {g.name for g in active_lib}]
        chosen = combined + custom_frags
    else:
        chosen = active_lib + custom_frags

    # Weights
    tot = sum(st.session_state.weights.values()) or 1
    weights = {k: v / tot for k, v in st.session_state.weights.items()}

    # Avoid options
    avoid_opts = {
        "nitro": st.session_state.avoid_nitro,
        "aldehyde": st.session_state.avoid_aldehyde,
        "reactive_acylhalide": st.session_state.avoid_reactive,
        "azide": st.session_state.avoid_toxic,
        "michael_acceptor": st.session_state.avoid_reactive,
        "epoxide": st.session_state.avoid_toxic,
    }

    col_info, col_run = st.columns([2, 1])
    with col_info:
        st.markdown(
            f"**Sites:** {valid_sites} | "
            f"**Mode:** {'concerted' if concerted else 'individual'} | "
            f"**Pool:** {len(chosen)} fragments | "
            f"**Target:** {st.session_state.n_analogs} analogs"
        )

    with col_run:
        run = st.button("Generate analogs 🚀", type="primary")

    if run:
        if not chosen:
            st.error("Fragment pool is empty. Adjust category/risk filters in Step 3.")
            st.stop()
        with st.spinner(f"Enumerating analogs from {len(chosen)} fragments × {len(site_groups)} site group(s)..."):
            df = core.generate_analogs(
                mol,
                selected_atoms=list(selected),
                chosen_frags=chosen,
                site_groups=site_groups,
                weights=weights,
                avoid_opts=avoid_opts,
                max_MW=st.session_state.max_MW,
                max_analogs=st.session_state.n_analogs,
                rank_by=st.session_state.rank_by,
                parent_name=st.session_state.parent_name,
            )
        st.session_state.analogs_df = df
        if df.empty:
            st.error("No analogs survived the filters. Relax filters in Step 3B.")
        else:
            st.success(f"Generated **{len(df)}** analogs ✅")

    df = st.session_state.analogs_df
    if df is not None and not df.empty:
        st.divider()
        st.subheader(f"Results — {len(df)} analogs")

        # Property distribution
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("Median MW", f"{df.MW.median():.0f}")
        pcol2.metric("Median QED", f"{df.QED.median():.3f}")
        pcol3.metric("Median ESOL logS", f"{df.ESOL.median():.2f}")
        pcol4.metric("Median Tanimoto sim", f"{df.sim.median():.3f}")

        # Filter controls
        with st.expander("Filter results"):
            f1, f2, f3 = st.columns(3)
            mw_max = f1.slider("Max MW", 200.0, 900.0, float(df.MW.max()), step=10.0)
            qed_min = f2.slider("Min QED", 0.0, 1.0, 0.0, step=0.05)
            cat_filter = f3.multiselect(
                "Fragment category",
                options=sorted(df.fragment_category.unique()),
                default=sorted(df.fragment_category.unique()),
            )
        df_show = df[
            (df.MW <= mw_max) & (df.QED >= qed_min) & (df.fragment_category.isin(cat_filter))
        ]
        st.dataframe(df_show, use_container_width=True, height=400)

        # Grid depiction
        st.subheader("Structure grid (top 20)")
        mols_grid = [Chem.MolFromSmiles(s) for s in df_show.smiles.head(20)]
        legs = [f"{i+1}. {c}" for i, c in enumerate(df_show.change.head(20))]
        img = Draw.MolsToGridImage(
            mols_grid, legends=legs, molsPerRow=4, subImgSize=(260, 200), maxMols=20
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.image(buf.getvalue(), use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 5 · Docking & cIFP
# ---------------------------------------------------------------------------

elif page == "5 · Docking & cIFP":
    st.title("Step 5 · Docking & PLIP/cIFP Validation")

    df_analogs = st.session_state.analogs_df
    if df_analogs is None or df_analogs.empty:
        st.warning("Generate analogs in Step 4 first.")
        st.stop()

    acd_available = bool(shutil.which("acd"))
    obabel_available = bool(shutil.which("obabel"))

    if not acd_available or not obabel_available:
        st.error(
            "Anyone Can Dock (`acd`) or `obabel` not found in PATH. "
            "Install via `pip install anyonecandock` and `apt install openbabel`."
        )

    # 5A · Prepare ligand list
    st.subheader("5A · Docking ligand list")
    include_parent = st.checkbox("Include original parent ligand", value=True)

    work = get_work_dir()
    dock_in = work / "docking_inputs"
    dock_in.mkdir(parents=True, exist_ok=True)

    rows = []
    if include_parent:
        rows.append({"compound": "original_ligand", "smiles": st.session_state.parent_smiles})
    for i, r in df_analogs.iterrows():
        rows.append({"compound": f"{st.session_state.parent_name}_A{i+1}", "smiles": r.smiles})

    lig_df = pd.DataFrame(rows).drop_duplicates("smiles").reset_index(drop=True)
    st.session_state.docking_ligands = lig_df
    st.dataframe(lig_df, use_container_width=True, height=200)

    smi_path = dock_in / "compounds.smi"
    with open(smi_path, "w") as fh:
        for _, r in lig_df.iterrows():
            fh.write(f"{r.smiles}\t{r.compound}\n")
    st.caption(f"Ligand SMI file written: `{smi_path}`")

    # 5B · Run docking
    st.divider()
    st.subheader("5B · ACD Batch Docking")

    receptor_for_docking = st.session_state.receptor_path or st.session_state.complex_path
    if not receptor_for_docking:
        st.warning("No receptor defined. Go to Step 1B and load a receptor first.")
    else:
        st.success(f"Receptor: `{receptor_for_docking}`")

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        exhaustiveness = st.slider("Exhaustiveness", 1, 32, 8)
        num_poses = st.slider("Poses per ligand", 1, 20, 10)
        dock_ph = st.number_input("pH for protonation", value=7.4, step=0.1)
    with bcol2:
        box_x = st.number_input("Box X (Å)", value=16.0)
        box_y = st.number_input("Box Y (Å)", value=16.0)
        box_z = st.number_input("Box Z (Å)", value=16.0)
        dock_out_dir = str(work / "docking_out")

    if acd_available and obabel_available and receptor_for_docking:
        if st.button("Run ACD batch docking", type="primary", disabled=not acd_available):
            cmd = core.build_acd_batch_cmd(
                receptor=receptor_for_docking,
                ligands_smi=str(smi_path),
                output_dir=dock_out_dir,
                exhaustiveness=exhaustiveness,
                num_poses=num_poses,
                ph=dock_ph,
                box_x=box_x, box_y=box_y, box_z=box_z,
            )
            cmd_str = " ".join(shlex.quote(str(x)) for x in cmd)
            st.code(cmd_str, language="bash")
            with st.spinner("Docking in progress…"):
                rc, output = core.run_command(cmd, log_path=str(work / "acd_batch_log.txt"))
            if rc == 0:
                st.success("Docking completed ✅")
            else:
                st.error(f"ACD returned exit code {rc}")
            with st.expander("ACD output log"):
                st.text(output[-3000:])
    else:
        st.info("Install ACD and OpenBabel to enable docking. You can still download ligand SMILES for external docking.")

    # 5C · Parse results
    st.divider()
    st.subheader("5C · Parse & Score")

    dock_out = Path(dock_out_dir)
    if st.button("Parse docking results") and dock_out.exists():
        best = core.parse_acd_score_csvs(str(dock_out))
        if best:
            st.write("Best score row:", best)
        pose_sdf = core.find_pose_sdf(str(dock_out))
        if pose_sdf:
            st.success(f"Top pose SDF: `{pose_sdf}`")
        else:
            st.warning("No SDF pose files found in docking output directory.")

    # 5D · cIFP
    st.divider()
    st.subheader("5D · PLIP / Distance cIFP")

    protein_for_plip = st.session_state.protein_path
    if not protein_for_plip:
        st.info("Protein-only PDB not available (Step 1B receptor split). cIFP will be skipped.")
    else:
        cifp_cutoff = st.slider("Contact distance cutoff (Å)", 3.0, 8.0, 4.0, 0.5)
        prefer_plip = st.checkbox("Prefer PLIP (if installed)", value=True)

        if st.button("Run cIFP analysis"):
            plip_available = bool(shutil.which("plipcmd") or shutil.which("plip"))
            cifp_dir = work / "plip_cifp"
            complex_dir = cifp_dir / "complexes"
            cifp_dir.mkdir(parents=True, exist_ok=True)
            complex_dir.mkdir(parents=True, exist_ok=True)

            # Find pose PDB files
            pose_pdbs = list((work / "docking_out" / "selected_pose_pdbs").glob("*.pdb")) if (
                work / "docking_out" / "selected_pose_pdbs"
            ).exists() else []

            if not pose_pdbs:
                st.warning("No pose PDB files found. Run docking first.")
            else:
                rows_cifp = []
                all_features: set = set()
                for p in pose_pdbs[:30]:
                    compound = p.stem
                    complex_pdb = str(complex_dir / f"{compound}_complex.pdb")
                    core.combine_protein_ligand_pdb(protein_for_plip, str(p), complex_pdb)
                    feats = []
                    method = "distance_fallback"
                    if prefer_plip and plip_available:
                        xml_path, err = core.run_plip(complex_pdb, str(cifp_dir / "plip_out"), compound)
                        feats = core.parse_plip_xml(xml_path) if xml_path else []
                        method = "PLIP" if feats else "distance_fallback"
                    if not feats:
                        feats = core.distance_contact_cifp(complex_pdb, cutoff=cifp_cutoff)
                    all_features.update(feats)
                    rows_cifp.append({
                        "compound": compound, "cifp_method": method,
                        "n_interactions": len(feats), "cifp_features": ";".join(feats),
                    })

                cifp_df = pd.DataFrame(rows_cifp)
                st.session_state.cifp_results = cifp_df
                st.success(f"cIFP computed for {len(cifp_df)} complexes")
                st.dataframe(cifp_df, use_container_width=True)

    cifp_df = st.session_state.cifp_results
    if cifp_df is not None and len(cifp_df) > 0 and len(cifp_df.columns) > 3:
        st.subheader("cIFP interaction matrix")
        features = sorted({f for row in cifp_df.cifp_features for f in str(row).split(";") if f})
        mat_rows = []
        for _, r in cifp_df.iterrows():
            present = set(str(r.cifp_features).split(";"))
            row = {"compound": r.compound}
            for f in features:
                row[f] = 1 if f in present else 0
            mat_rows.append(row)
        if mat_rows:
            mat_df = pd.DataFrame(mat_rows)
            st.dataframe(mat_df.set_index("compound"), use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE Export
# ---------------------------------------------------------------------------

elif page == "📦 Export":
    st.title("📦 Export Results")

    df = st.session_state.analogs_df
    if df is None or df.empty:
        st.warning("No analogs generated yet. Complete Step 4 first.")
        st.stop()

    parent_name = st.session_state.parent_name or "parent"
    work = get_work_dir()

    # CSV download
    st.subheader("Analog table")
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download analogs CSV",
        data=csv_bytes,
        file_name=f"{parent_name}_analogs.csv",
        mime="text/csv",
    )

    # SMILES download
    smi_lines = "\n".join(f"{r.smiles}\t{parent_name}_A{i+1}" for i, r in df.iterrows())
    st.download_button(
        "⬇️ Download analogs SMI",
        data=smi_lines.encode(),
        file_name=f"{parent_name}_analogs.smi",
        mime="text/plain",
    )

    # 3D SDF generation
    st.divider()
    st.subheader("3D ligand files (SDF / PDB)")
    formats_sel = st.multiselect(
        "Output formats", ["SDF", "PDB", "MOL2"], default=["SDF"]
    )
    mmff_opt = st.checkbox("MMFF geometry optimisation", value=True)

    if st.button("Generate 3D ligands"):
        lig_table = df[["smiles"]].copy()
        lig_table["compound"] = [f"{parent_name}_A{i+1}" for i in range(len(df))]
        out_dir = work / "ligands_3d"
        with st.spinner("Generating 3D conformers…"):
            manifest = core.generate_3d_ligand_files(lig_table, out_dir, formats=formats_sel, mmff=mmff_opt)
        ok = manifest[manifest.status == "ok"]
        st.success(f"3D files generated: {len(ok)}/{len(manifest)}")
        st.dataframe(manifest, use_container_width=True)

        # Combined SDF download
        combined = out_dir / "all_ligands_3d.sdf"
        if combined.exists():
            st.download_button(
                "⬇️ Download combined SDF",
                data=combined.read_bytes(),
                file_name=f"{parent_name}_3d.sdf",
                mime="chemical/x-mdl-sdfile",
            )

    # ZIP all outputs
    st.divider()
    st.subheader("Full results ZIP")
    if st.button("Build ZIP archive"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            # Analog CSV
            z.writestr(f"{parent_name}_analogs.csv", df.to_csv(index=False))
            z.writestr(f"{parent_name}_analogs.smi", smi_lines)
            # 3D files if they exist
            lig3d = work / "ligands_3d"
            if lig3d.exists():
                for f in lig3d.glob("**/*.*"):
                    z.write(f, arcname=f"ligands_3d/{f.name}")
            # cIFP if available
            cifp_df = st.session_state.cifp_results
            if cifp_df is not None and not cifp_df.empty:
                z.writestr("cifp_results.csv", cifp_df.to_csv(index=False))
            # Docking outputs if present
            dock_out = work / "docking_out"
            if dock_out.exists():
                for f in dock_out.rglob("*.*"):
                    if f.is_file():
                        z.write(f, arcname=f"docking_out/{f.relative_to(dock_out)}")
            # Session snapshot
            snap = {
                "parent_smiles": st.session_state.parent_smiles,
                "parent_name": parent_name,
                "design_workflow": st.session_state.design_workflow,
                "selected_atoms": sorted(st.session_state.selected_atoms),
                "n_analogs_generated": len(df),
                "rank_by": st.session_state.rank_by,
            }
            z.writestr("session_info.json", json.dumps(snap, indent=2))

        buf.seek(0)
        st.download_button(
            "⬇️ Download full ZIP",
            data=buf.getvalue(),
            file_name=f"{parent_name}_analog_designer_results.zip",
            mime="application/zip",
        )

    # Quick stats
    st.divider()
    st.subheader("Session summary")
    st.json({
        "parent_smiles": st.session_state.parent_smiles,
        "parent_name": parent_name,
        "design_workflow": st.session_state.design_workflow,
        "selected_atoms": sorted(st.session_state.selected_atoms),
        "analogs_generated": len(df) if df is not None else 0,
        "fragment_categories": dict(
            Counter(df.fragment_category.tolist()) if df is not None else {}
        ),
        "receptor_loaded": bool(st.session_state.receptor_path),
        "docking_run": st.session_state.docking_ligands is not None,
        "cifp_computed": st.session_state.cifp_results is not None,
    })
