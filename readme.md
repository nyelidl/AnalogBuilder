# AnalogBuilder

**AnalogBuilder** is a Colab-based workflow for practical analog generation and early-stage computational prioritization. It supports both **ligand-guided** and **structure-guided** design, with optional docking, interaction profiling, and 3D ligand export.

This notebook is intended as a lightweight research workflow, not a full drug-discovery platform. It helps generate and organize candidate derivatives/analogs for follow-up docking, visualization, and downstream analysis.

---

## Key Features

- **Ligand-guided analog generation**
  - Generate derivatives from a parent SMILES.
  - Select one or more modification sites directly on the molecular structure.
  - Support individual one-position modification or concerted multi-position modification.

- **Structure-guided analog generation**
  - Use a protein–ligand complex to guide functional-group suggestions.
  - Detect nearby pocket residues around the ligand.
  - Suggest fragment types based on local pocket properties such as hydrogen-bonding, hydrophobic, aromatic, acidic, and basic residues.

- **Fragment-based derivative construction**
  - Use built-in fragment libraries or user-defined custom fragments.
  - Optional pocket-guided fragment blending.
  - Basic property filters for molecular weight, synthetic accessibility, toxic/reactive alerts, and formal charge.

- **pH/charge annotation**
  - Supports pKaNET-style charge annotation when available.
  - Can annotate charge states without deleting all generated analogs.
  - Useful for early triage before docking or MD preparation.

- **Docking-ready output**
  - Prepares ligand lists for Anyone Can Dock / AutoDock Vina workflows.
  - Supports original ligand plus all generated analogs.
  - Keeps full protein–ligand complex input for automatic docking-center detection when available.

- **Interaction analysis support**
  - Designed to work with PLIP/cIFP-style interaction profiling after docking.
  - Exports summary tables for docking scores, pose information, and interaction fingerprints when available.

- **3D ligand export**
  - Generates minimized 3D ligand structures using RDKit ETKDG and MMFF/UFF fallback.
  - Exports SDF, PDB, and optionally MOL2 files if Open Babel is available.
  - Packages final results into a downloadable ZIP archive.

---

## Workflow Overview

```text
1. Define parent ligand
   └── Input parent SMILES and basic design settings

2. Select modification sites
   └── Interactive atom picker or manual atom-index input

3. Choose design mode
   ├── Ligand-guided fragment selection
   └── Structure-guided pocket analysis, if a complex is provided

4. Generate analogs
   └── Build candidate derivatives and rank by simple property-based heuristics

5. Dock and analyze
   ├── Prepare receptor/complex and ligand list
   ├── Run Anyone Can Dock batch docking
   ├── Parse docking output
   ├── Run optional PLIP/cIFP interaction analysis
   └── Export dashboard, CSV files, docking files, and 3D ligand structures
```

---

## Installation

AnalogBuilder is designed primarily for **Google Colab**. The notebook can install or use the following tools depending on which modules are enabled:

- RDKit
- pandas / NumPy / matplotlib
- Open Babel
- Anyone Can Dock
- PLIP, optional
- fpocket, optional
- py3Dmol, optional
- pKaNET, optional

A typical Colab setup cell should install the required dependencies before running the workflow.

---

## Basic Usage

1. Open the notebook in Google Colab.
2. Run the setup/install cell.
3. Enter a parent ligand SMILES.
4. Select atom positions for derivatization.
5. Choose ligand-based or structure-based guidance.
6. Generate analogs.
7. Optionally run docking and interaction analysis.
8. Download the final ZIP archive containing tables, structures, plots, and dashboard files.

---

## Input Options

### Ligand-based mode

Use this mode when you only have a parent ligand structure or SMILES.

Typical inputs:

- Parent SMILES
- Selected atom indices for modification
- Fragment category preferences
- Number of analogs to generate

### Structure-based mode

Use this mode when you have a protein–ligand complex or a receptor structure with a known ligand pose.

Typical inputs:

- Protein–ligand complex PDB/CIF, or PDB ID
- Parent ligand SMILES
- Optional ligand residue name
- Docking center mode, such as automatic center detection from a reference ligand

---

## Output Files

AnalogBuilder can generate:

- `*_analogs.csv` — generated analog table
- `*_analogs.smi` — SMILES list for generated analogs
- `compounds_for_acd.smi` — docking input ligand list
- `final_docking_cifp_summary.csv` — final docking/interaction summary
- `dashboard.html` — HTML dashboard
- `ligand_3d_manifest.csv` — list of generated 3D ligand files
- `all_ligands_3d.sdf` — combined 3D ligand SDF
- individual ligand `.sdf`, `.pdb`, and optional `.mol2` files
- final ZIP archive containing results and metadata

---

## Notes and Limitations

- AnalogBuilder performs **early-stage computational prioritization**, not experimental validation.
- Docking scores and interaction fingerprints should be interpreted as screening-level evidence.
- Generated analogs may require chemical review before synthesis or further simulation.
- pH/charge annotation can be useful, but it should not be treated as definitive without additional validation.
- Structure-guided suggestions depend on the quality of the input complex or docked parent pose.
- The workflow is designed to assist hypothesis generation and compound triage, not to guarantee binding affinity or biological activity.

---

## Suggested Citation / Acknowledgement

If you use or adapt this workflow, please cite or acknowledge the relevant tools used in your run, such as RDKit, Anyone Can Dock, AutoDock Vina, Open Babel, PLIP, fpocket, and pKaNET when applicable.

---

## License

Add your preferred license here, for example:

- MIT License
- Apache License 2.0
- BSD 3-Clause License

---

## Project Status

AnalogBuilder is under active development. Interfaces, defaults, and optional modules may change as the workflow is refined.
