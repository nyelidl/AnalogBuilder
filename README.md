# Analog Builder — Streamlit App

Interactive ligand analog generation with optional structure-based docking and PLIP/cIFP analysis.

## Project layout

```
cody_app/
├── app.py          # Streamlit UI — all pages and widgets
├── core.py         # Chemistry / computation backend (no UI imports)
├── requirements.txt
└── README.md
```

## Running locally

```bash
pip install -r requirements.txt
# Optional extras (needed for docking and structure-based features):
# pip install anyonecandock pkanet-cloud[recommended] plip gemmi
# apt install openbabel fpocket          # Ubuntu / Colab

streamlit run app.py
```

## Workflow

| Page | What it does |
|------|-------------|
| **1 · Parent Ligand** | Paste SMILES, choose ligand-based or structure-based workflow, optionally download/upload receptor |
| **2 · Atom Selection** | Pick attachment atoms; toggle concerted vs individual mode |
| **3 · Design Settings** | Goal weights, fragment category filters, risk level, avoid options, pocket residue guidance |
| **4 · Generate Analogs** | Enumerate analogs, filter, rank, view structure grid |
| **5 · Docking & cIFP** | Run ACD batch docking, parse scores, compute PLIP/distance cIFP interaction fingerprints |
| **📦 Export** | Download CSV, SMI, 3D SDF, and full ZIP archive |

## Architecture

### `core.py` (no Streamlit imports)
- Fragment library (`Frag` dataclass + `LIBRARY`)
- Property calculators: `sa_score`, `esol_logS`, `morgan`
- Analog generation: `attach`, `attach_to_sites`, `generate_analogs`
- Pocket analysis: `analyze_complex_distance_shell`, `suggest_fragments_from_residues`
- PDB helpers: `download_pdb`, `split_protein_ligand`, `combine_protein_ligand_pdb`
- ACD command builders: `build_acd_dock_cmd`, `build_acd_batch_cmd`
- Score parsing: `parse_acd_score_csvs`, `find_pose_sdf`
- cIFP: `distance_contact_cifp`, `run_plip`, `parse_plip_xml`
- 3D output: `build_3d_mol`, `generate_3d_ligand_files`

### `app.py` (Streamlit only)
- Session state management (`st.session_state`)
- Page routing via sidebar radio
- Calls `core.*` functions; never does chemistry itself
