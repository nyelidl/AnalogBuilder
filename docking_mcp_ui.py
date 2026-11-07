"""Step 5 remote docking controls, separate from the legacy CLI renderer."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from acd_mcp import (ACDMCPClient, MCPError, SubmissionUncertain, TERMINAL_STATUSES,
                     build_payload, load_receipt, payload_fingerprint, public_url,
                     resolve_center, result_rows, write_receipt)


def _client():
    endpoint = os.environ.get("ACD_MCP_URL", "")
    if st.session_state.get("_mcp_client_endpoint") != endpoint:
        st.session_state.pop("_mcp_client", None)
    if "_mcp_client" not in st.session_state:
        st.session_state["_mcp_client"] = ACDMCPClient()
        st.session_state["_mcp_client_endpoint"] = endpoint
    return st.session_state["_mcp_client"]


def _link(label, value):
    url = public_url(value)
    if url:
        st.link_button(label, url)


def _render_job(job, receipt, out_dir):
    status = job.get("status", "unknown")
    st.write(f"Job: `{receipt['job_id']}` · **{status}**")
    if job.get("message"):
        st.write(job["message"])
    request = receipt.get("request", {})
    if request:
        st.caption(f"Requested pH {request['ph']:g}; {request['protonation_mode']}; "
                   f"exhaustiveness {request['exhaustiveness']}; {request['num_modes']} poses.")
    else:
        st.caption("Recovered job: original pH and preparation settings are not available in this receipt.")
    receptor = job.get("receptor", {})
    if receptor:
        st.markdown("**Grid reported by the docking service**")
        st.json({k: receptor.get(k) for k in ("center", "size", "cocrystal_ligand_id")})
        if request.get("grid"):
            mismatches = []
            for group, prefix in (("center", "center"), ("size", "size")):
                for axis in "xyz":
                    actual = receptor.get(group, {}).get(axis)
                    expected = request["grid"].get(f"{prefix}_{axis}")
                    if actual is not None and expected is not None and abs(actual - expected) > 1e-3:
                        mismatches.append(f"{group} {axis}")
            if mismatches:
                st.warning("The service reported a grid different from this receipt's request: "
                           + ", ".join(mismatches) + ". Review it before comparing scores.")
        if receptor.get("reference_warning"):
            st.warning(receptor["reference_warning"])
    rows = result_rows(job)
    if rows:
        st.session_state["dock_results"] = rows
        table = pd.DataFrame(rows)
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Docking results CSV", table.to_csv(index=False).encode(),
                           file_name=f"{receipt['job_id']}_scores.csv", mime="text/csv")
        try:
            (out_dir / f"{receipt['job_id']}_scores.csv").write_text(table.to_csv(index=False))
        except OSError as exc:
            st.warning(f"Remote results are available, but their local CSV could not be saved: {exc}")
        for item in job.get("results", []):
            name = item.get("name", "unnamed")
            with st.expander(f"{name} — {item.get('status', 'unknown')}"):
                if item.get("status") != "ok":
                    st.error(item.get("error") or "The service returned no successful result for this ligand.")
                    st.json(item)
                    continue
                for warning in ("protonation_fallback", "pose_selection_warning", "structure_warning"):
                    if item.get(warning):
                        st.warning(item[warning])
                st.write("Prepared SMILES:", item.get("prepared_smiles", "unavailable"))
                st.write("Pose selection:", item.get("pose_selection_method", "unavailable"))
                if item.get("scores"):
                    st.dataframe(pd.DataFrame(item["scores"]), hide_index=True)
                    st.caption("rmsd_lb / rmsd_ub are Vina pose-to-pose bounds, not co-crystal RMSD.")
                _link("Open 3D viewer", item.get("viewer_3d_url"))
                _link("Selected pose SDF", item.get("selected_pose_sdf_url"))
                _link("Selected pose PDB", item.get("selected_pose_pdb_url"))
                _link("Protein–ligand complex", item.get("complex_pdb_url"))
                _link("2D interaction diagram", item.get("two_d_interaction_png_url"))
                source = item.get("interaction_analysis_source", "")
                if source:
                    st.caption(source)
                if item.get("interaction_analysis_warning"):
                    st.caption(item["interaction_analysis_warning"])
                if item.get("interactions"):
                    st.dataframe(pd.DataFrame(item["interactions"]), hide_index=True)
                elif source == "not_configured":
                    st.info("Interaction classification was not run by the service.")
    if status == "failed":
        st.error(job.get("error") or job.get("message") or "Remote docking failed.")
        with st.expander("Service failure details"):
            st.json(job)
    _link("Download all remote files and logs (ZIP)", job.get("download_url"))


def render_mcp_docking(receptor, reference, lig_df, out_dir, box, ph, exhaustiveness, num_modes):
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        st.error(f"Cannot save docking job receipts in the output directory: {exc}")
        return
    st.markdown("### Dock with Anyone Can Dock MCP")
    st.caption("Your receptor structure and ligand SMILES are sent to the Anyone Can Dock service. "
               "Docking runs remotely; this mode does not require local ACD or Open Babel.")
    # The service owns preparation; its MCP/API version is not the pinned local CLI version.
    st.caption("Remote preparation can differ from the local CLI. Review prepared SMILES, "
               "warnings and the returned logs. Vina scores are computational estimates, "
               "not measured binding affinities.")
    with st.expander("Remote preparation settings"):
        protonation = st.selectbox("Ligand protonation", ["pkanet", "neutral", "dimorphite"], key="mcp_protonation")
        pubchem = st.checkbox("Use PubChem pKa evidence", value=True, key="mcp_pubchem")
    payload = None
    try:
        if not receptor:
            raise ValueError("Load a receptor above before submitting a new job.")
        receptor_text = Path(receptor).read_text()
        reference_text = Path(reference).read_text() if reference and Path(reference).is_file() else None
        center = resolve_center(receptor_text, reference_text,
                                st.session_state.get("dock_center_mode", "Protein centre"),
                                tuple(st.session_state.get(f"dock_man_c{axis}", 0.0) for axis in "xyz"))
        limit = int(os.environ.get("ACD_MCP_MAX_LIGANDS", "100"))
        if len(lig_df) > limit:
            raise ValueError(f"This deployment allows {limit} ligands per remote job; "
                             "reduce the selected set or ask the host to adjust ACD_MCP_MAX_LIGANDS.")
        payload = build_payload(receptor_text,
            [{"name": str(r.compound), "smiles": str(r.smiles)} for r in lig_df.itertuples()],
            center, box, ph, exhaustiveness, num_modes, protonation, pubchem, Path(receptor).stem)
        st.caption(f"Sending {len(lig_df)} ligands · centre {tuple(round(v, 3) for v in center)} Å "
                   f"· box {tuple(float(v) for v in box)} Å · pH {ph:g}")
    except (OSError, ValueError, KeyError) as exc:
        st.info(str(exc))
    fingerprint = payload_fingerprint(payload) if payload else "recovery"
    receipt_path = out_dir / f"mcp_{fingerprint}.json"
    invalid_receipt = False
    try:
        receipt = load_receipt(receipt_path)
    except MCPError as exc:
        st.error(str(exc))
        invalid_receipt = True
        receipt = None

    # Recovery remains available even without local structures or a previous websocket.
    with st.expander("Recover a remote job after a refresh"):
        recover_id = st.text_input("Saved job ID", key="mcp_recover_id")
        if st.button("Recover job", disabled=not recover_id.strip()):
            try:
                job = _client().job(recover_id.strip())
                if job.get("job_id") != recover_id.strip():
                    raise MCPError("The service returned a different job ID.")
                receipt = {"job_id": recover_id.strip(), "status": job.get("status"),
                           "submission_state": "accepted", "job": job, "recovered": True}
                write_receipt(receipt_path, receipt)
                invalid_receipt = False
                st.session_state["_mcp_paused"] = False
            except (MCPError, ValueError, OSError) as exc:
                st.error(str(exc))
                st.session_state.pop("_mcp_client", None)

    confirmed_health = st.session_state.get("_mcp_health", {})
    if st.button("Check remote docking service"):
        try:
            health = _client().call("pingServer", {})
            st.session_state["_mcp_health"] = confirmed_health = health
            if not all(health.get(k) for k in ("ok", "vina_available", "openbabel_available")):
                st.error("The service is reachable but its docking tools are not ready.")
        except MCPError as exc:
            st.error(str(exc))
            st.session_state["_mcp_health"] = confirmed_health = {}
            st.session_state.pop("_mcp_client", None)
    ready = all(confirmed_health.get(k) for k in ("ok", "vina_available", "openbabel_available"))
    if ready:
        st.success("Remote Vina and Open Babel are available.")
    else:
        st.info("Check the remote service before submitting.")

    uncertain = bool(receipt and receipt.get("submission_state") in {"submitting", "uncertain"})
    active = bool(receipt and receipt.get("job_id") and receipt.get("status") not in TERMINAL_STATUSES)
    new_run = False
    if uncertain:
        st.warning("A previous submission has no confirmed job ID. It may still be running remotely. "
                   "Recover its ID before submitting again.")
        new_run = st.checkbox("I understand a new submission may create a duplicate job", key="mcp_duplicate_ack")
    elif receipt and receipt.get("job_id") and not active:
        new_run = st.checkbox("Create another job with these settings", key="mcp_new_job")
    blocked = invalid_receipt or active or (bool(receipt and receipt.get("job_id")) and not new_run) or (uncertain and not new_run)
    if st.button("Submit remote docking", type="primary", disabled=not ready or payload is None or blocked):
        request = {k: payload[k] for k in ("grid", "ph", "exhaustiveness", "num_modes", "protonation_mode", "use_pubchem")}
        receipt = {"fingerprint": fingerprint, "submission_state": "submitting", "timestamp": time.time(),
                   "request": request, "ligands": payload["ligands"],
                   "service": {"api": confirmed_health.get("api"),
                               "api_version": confirmed_health.get("version"),
                               "mcp_server": {k: _client().server_info.get(k) for k in ("name", "version")}}}
        # Persist intent before sending so reruns cannot silently duplicate an uncertain request.
        write_receipt(receipt_path, receipt)
        try:
            response = _client().submit(payload)
            receipt.update(job_id=response["job_id"], status=response.get("status", "queued"),
                           submission_state="accepted", submission=response)
            st.session_state["_mcp_paused"] = False
        except SubmissionUncertain as exc:
            receipt.update(submission_state="uncertain", error=str(exc))
            st.error(str(exc))
        except MCPError as exc:
            receipt.update(submission_state="not_submitted", error=str(exc))
            st.error(str(exc))
            st.session_state.pop("_mcp_client", None)
        finally:
            write_receipt(receipt_path, receipt)
        st.rerun()

    if not receipt or not receipt.get("job_id"):
        return
    st.download_button("Save job receipt (JSON)", json.dumps(receipt, indent=2),
                       file_name=f"{receipt['job_id']}_receipt.json", mime="application/json")
    paused = st.checkbox("Stop refreshing this job", key="_mcp_paused")
    if paused:
        st.info("Only status refreshing is paused. The remote job continues; the service exposes no cancel operation.")
    interval = "5s" if receipt.get("status") not in TERMINAL_STATUSES and not paused else None

    @st.fragment(run_every=interval)
    def job_status():
        current = load_receipt(receipt_path) or receipt
        refresh = st.button("Refresh job status", key="mcp_refresh")
        if refresh or (not paused and (current.get("status") not in TERMINAL_STATUSES or not current.get("job"))):
            try:
                job = _client().job(current["job_id"])
                if job.get("job_id") != current["job_id"]:
                    raise MCPError("The service returned a different job ID.")
                previous_status = current.get("status")
                current.update(status=job.get("status", "unknown"), job=job, updated=time.time())
                write_receipt(receipt_path, current)
                if previous_status not in TERMINAL_STATUSES and current["status"] in TERMINAL_STATUSES:
                    st.rerun()  # stop the polling timer once the job reaches a terminal state
            except (MCPError, ValueError, OSError) as exc:
                st.error(f"Status unavailable; keep job ID {current['job_id']}: {exc}")
                st.session_state.pop("_mcp_client", None)
        _render_job(current.get("job", current.get("submission", {})), current, out_dir)

    job_status()
