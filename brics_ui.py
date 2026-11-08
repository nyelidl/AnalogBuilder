"""Reviewable BRICS cuts for Step 2 and generation controls for Step 4."""
from __future__ import annotations

import hashlib
import json

from rdkit import Chem
import streamlit as st

from brics_editor import (RANK_OPTIONS, cap_with_hydrogen, draw_edit_svg, find_brics_edits)

EDIT_MODES = ["Append fragments", "BRICS trim", "BRICS replace"]


def parent_key(parent):
    # Noncanonical ordering ties the selected cut to the parent's displayed atom indices.
    return hashlib.sha256(Chem.MolToSmiles(parent, canonical=False, isomericSmiles=True).encode()).hexdigest()


def _label(edit):
    return (f"Cut {edit.atom_a}–{edit.atom_b}; retain atom {edit.keep_anchor} side "
            f"({edit.retained.GetNumHeavyAtoms()} heavy atoms), "
            f"remove {edit.removed.GetNumHeavyAtoms()} heavy atoms")


def render_cut_picker(parent):
    try:
        edits = find_brics_edits(parent)
    except ValueError as exc:
        st.error(str(exc))
        return False
    key = parent_key(parent)
    if st.session_state.get("_brics_parent") != key:
        st.session_state["_brics_parent"] = key
        st.session_state.pop("_brics_edit", None)
        st.session_state.analogs_df = None
    if not edits:
        st.info("No supported single BRICS bond is present. Use Append fragments or choose another parent. "
                "Ring bonds and double-bond cuts are not offered in this version.")
        return False
    by_key = {e.key: e for e in edits}
    options = list(by_key)
    previous = st.session_state.get("_brics_edit")
    chosen = st.selectbox("BRICS cut and retained side", options,
                          index=options.index(previous) if previous in options else 0,
                          format_func=lambda k: _label(by_key[k]), key=f"brics_cut_{key}")
    if chosen != previous:
        st.session_state.analogs_df = None
    st.session_state["_brics_edit"] = chosen
    st.session_state.selected_atoms = set()  # BRICS uses bonds, not append-mode H sites.
    edit = by_key[chosen]
    st.image(draw_edit_svg(parent, edit), width="stretch")
    st.caption("Green = retained piece · orange = removed piece · red = cut bond. "
               "Each product changes this one bond; ring opening and two-point linker replacement are not included.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Retained BRICS fragment**")
        st.code(Chem.MolToSmiles(edit.retained, isomericSmiles=True), language=None)
    with right:
        st.markdown("**Removed BRICS fragment**")
        st.code(Chem.MolToSmiles(edit.removed, isomericSmiles=True), language=None)
    st.markdown("**Trim preview (retained piece capped with hydrogen)**")
    st.code(Chem.MolToSmiles(cap_with_hydrogen(edit.retained), isomericSmiles=True), language=None)
    st.caption("Trim replaces the cut attachment with H. Replace joins a library [*]-fragment "
               "and checks the new junction against RDKit BRICS rules. Compatibility does not prove synthesizability.")
    return True


def selected_edit(parent):
    if st.session_state.get("_brics_parent") != parent_key(parent):
        raise ValueError("The parent changed. Select and review a new BRICS cut in Step 2.")
    edits = {e.key: e for e in find_brics_edits(parent)}
    choice = st.session_state.get("_brics_edit")
    if choice not in edits:
        raise ValueError("Select and review a BRICS cut in Step 2 before generating.")
    return edits[choice]


def render_generation_controls(parent, operation):
    st.markdown(f"### {operation}")
    try:
        edit = selected_edit(parent)
    except ValueError as exc:
        st.warning(str(exc))
        if st.button("Choose BRICS cut in Step 2"):
            st.session_state.step = 2
            st.rerun()
        st.stop()
    st.write(_label(edit))
    with st.expander("Review selected cut"):
        st.image(draw_edit_svg(parent, edit), width="stretch")
    choices = list(RANK_OPTIONS)
    current = st.session_state.get("_brics_rank", choices[0])
    rank = st.selectbox("Rank BRICS products by", choices,
                        index=choices.index(current) if current in choices else 0, key="brics_rank_widget")
    st.session_state["_brics_rank"] = rank
    cap = 1
    if operation == "BRICS replace":
        cap = st.number_input("Maximum replacement attempts", min_value=1, max_value=5000,
                              value=int(st.session_state.get("_brics_attempts", 5000)), step=100,
                              key="brics_attempt_widget")
        st.session_state["_brics_attempts"] = int(cap)
        st.caption("Uses the selected fragment categories, size limit and custom pool. "
                   "Generic [*] fragments are checked at the actual product junction; incompatible candidates are skipped.")
    st.caption("Ranks by the property selected here. Append-mode goal weights do not apply to BRICS edits. "
               "Molecular weight, formal-charge and selected structural-alert filters still apply.")
    return edit, rank, int(cap)


def generation_key(parent, edit, operation, fragments, rank, cap, max_mw, max_analogs, avoid_opts):
    content = {"parent": parent_key(parent), "edit": edit.key, "operation": operation,
               "fragments": [(f.name, f.smiles, f.category) for f in fragments] if operation == "BRICS replace" else [],
               "rank": rank, "cap": cap, "max_mw": max_mw, "max_analogs": max_analogs,
               "avoid_opts": avoid_opts}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
