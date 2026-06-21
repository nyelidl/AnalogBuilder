<img src="https://github.com/nyelidl/AnalogBuilder/blob/main/.fig/AB.svg" alt="Analog Builder logo" width="120"/>

# Analog Designer

**Analog Designer** is a Streamlit web application for ML-guided drug analog generation and structure-based computational prioritisation. It integrates pocket analysis, ChemBERTa-powered fragment ranking, molecular docking, protein–ligand interaction profiling, and 3D visualisation in a single browser-deployable interface — no installation, no GPU, no computational chemistry expertise required.

> *21 Jun 2026 · Kowit Hengphasatporn · kowit@ccs.tsukuba.ac.jp · CCS, University of Tsukuba*

---

## What's New in This Version

- **ChemBERTa ML fragment ranking** — transformer-based pocket-aware scoring (zero-shot, no fine-tuning)
- **Pocket void analysis** — 3D sub-pocket detection with fragment size recommendation and interactive viewer
- **PLIP interaction analysis** — 8 interaction types, design recommendations, cIFP parent vs analog comparison
- **Structure-based Mode A/B** — auto-detect co-crystal vs apo PDB, skip docking when complex is available
- **Batch score plot** — ACD-style scatter plot with RMSD ring encoding
- **Per-pose RMSD vs co-crystal** — MCS-based heavy-atom RMSD for all docked poses
- **Paginated structure grid** — 50 compounds/page with page number input

---

## Key Features

### 🧬 Ligand-based analog generation
- Enter a parent SMILES or draw it in the Ketcher 2D editor
- Click attachment atoms on the 2D structure — multiple sites supported
- Optional: enter binding-site residues to activate ML-guided fragment ranking
- Generate up to 1,000+ analogs with property filters

### 🔬 Structure-based analog generation
- Load a protein PDB from RCSB search, PDB ID, or file upload
- **Auto-detect**: if PDB contains a co-crystal ligand → extract SMILES + skip docking
- If apo PDB → ACD docks your ligand automatically (Mode B)
- If complex detected: choose to use it as-is or dock your own ligand instead

### 🤖 ML-Guided Fragment Ranking (ChemBERTa)
See the [ML section](#ml-model--pocket-aware-fragment-ranking) below for full details.

### 📐 Pocket Void Analysis
- Measures unoccupied 3D space between ligand surface and pocket residues
- Clusters growth residues into spatial sub-pockets
- Classifies each sub-pocket as small / medium / large / extended
- Filters fragment pool by size class to avoid steric clashes
- **3D pocket viewer** — colour-coded by sub-pocket with interactive residue labels

### 🔬 PLIP Interaction Analysis
- Runs PLIP on the complex after docking or from co-crystal
- Detects 8 interaction types: HBOND, HYDROPHOBIC, PISTACK, PICATION, SALTBRIDGE, HALOGEN, WATERBRIDGE, METAL
- Key interactions (HBOND, SALTBRIDGE, PICATION, METAL) are flagged — loss triggers warning
- Design recommendation: 🔒 preserve list + 🌱 growth vector list
- cIFP Tanimoto comparison: parent vs each analog after docking

### 🚀 Docking (ACD / AutoDock Vina)
- AnyonCanDock v1.1.3 with AutoDock Vina 1.2.7 (auto-downloaded)
- pH-aware protonation via dimorphite-DL
- Batch score plot (ACD-style): binding energy vs compound, RMSD ring encoding
- Per-pose table: affinity (kcal/mol) + RMSD vs co-crystal (Å) with MCS alignment
- ✅ RMSD ≤ 2 Å / ⚠️ 2–3 Å / ❌ > 3 Å

### 🧊 3D Visualisation (py3Dmol)
- **Step 3** — parent ligand in pocket immediately after complex is loaded
- **Step 5** — compound selector to view any docked analog pose
- Protein: cartoon, spectrum colour, 45% opacity
- Ligand (cyan), reference (magenta), pocket residues (orange)
- Dark/light mode aware

---

## ML Model — Pocket-Aware Fragment Ranking

This is the core novel contribution of Analog Designer.

### Model: ChemBERTa

| Property | Value |
|----------|-------|
| Model | `seyonec/ChemBERTa-zinc-base-v1` |
| Architecture | RoBERTa transformer |
| Pre-training data | 77 million SMILES (ZINC database) |
| Pre-training task | Masked language modelling on SMILES |
| Output dimension | 384-dimensional embedding vectors |
| Inference | CPU-only, ~0.3–0.5 s for 76 fragments |
| Fine-tuning required | ❌ None — zero-shot |
| GPU required | ❌ None |

### How It Works

```
Input: binding-site residues (from PLIP or user input)
          ↓
1. Map residue property tags → context SMILES
   e.g. hydrophobic pocket → ["CC1CCCCC1", "c1ccccc1", ...]
        acidic pocket      → ["CCN(CC)CC", "C1CCNCC1", ...]

2. Embed with ChemBERTa (mean-pool over tokens → L2-normalise)
   v_pocket ∈ ℝ³⁸⁴

3. Embed each fragment SMILES the same way
   v_frag_i ∈ ℝ³⁸⁴

4. Cosine similarity → shift to [0, 1]
   score_CB(i) = (v_pocket · v_frag_i + 1) / 2

5. Blend with rule-based co-occurrence score (α = 0.7 default)
   score_final(i) = 0.7 × score_CB(i) + 0.3 × score_RB(i)
```

### Rule-Based Fallback

When `transformers` or `torch` are not installed, the system automatically falls back to a curated **10 × 8 co-occurrence matrix** (pocket property tag × fragment category) derived from published PDB analyses. No error, no user action needed — the UI shows 🟢 ChemBERTa or 🟡 Rule-based fallback.

### Fragment Library

76 curated fragments across 8 medicinal chemistry categories:

| Category | Count | Examples |
|----------|------:|---------|
| Aromatic | 16 | phenyl, pyridin-3-yl, thiophen-2-yl, indol-3-yl |
| Polar | 13 | hydroxyl, methoxy, methylsulfonyl, cyano |
| Hydrophobic | 12 | cyclopropyl, tert-butyl, cyclohexyl |
| Basic | 9 | piperidine, morpholine, N-methylpiperazine |
| Halogen | 8 | fluoro, chloro, trifluoromethyl |
| Bioisostere | 7 | oxetan-3-yl, bicyclo[1.1.1]pentan-1-yl |
| Acidic | 6 | carboxyl, sulfonamide, tetrazole |
| Solubility | 5 | methoxyethyl, hydroxymethyl |

All fragments: MW ≤ 250 Da · rotatable bonds ≤ 6 · clogP ∈ [−3, 4.5] · exactly one `[*]` attachment point.

---

## Workflow — 6 Steps

```
Step 1A  Load protein structure (structure track)
         └── RCSB search / PDB ID / file upload
         └── Auto-detect: complex → skip docking | apo → dock automatically

Step 1B  Define parent compound
         └── SMILES input or Ketcher 2D editor

Step 2   Select attachment atoms
         └── Click atoms on 2D depiction (multiple sites allowed)

Step 3   Pocket analysis + ML fragment ranking        ← ML runs here
         ├── (Structure track) ACD docking if apo PDB (Mode B)
         ├── PLIP interaction analysis → 8 interaction types
         ├── ChemBERTa/rule-based fragment ranking → ranked table
         ├── Void analysis → sub-pocket 3D viewer + size filter
         └── Design recommendation: 🔒 preserve / 🌱 grow

Step 4   Generate analogs
         └── Enumerate fragments × attachment atoms
         └── Filter: MW, QED, reactive groups, clogP
         └── Structure grid (50/page)

Step 5   Docking + Evaluation
         ├── ACD / AutoDock Vina 1.2.7
         ├── Batch score plot (BE vs compound, RMSD ring colour)
         ├── Per-pose table: affinity + RMSD vs co-crystal
         ├── cIFP Tanimoto: parent vs analogs
         └── 3D viewer: select any analog pose

Step 6   Export
         └── SMILES CSV · docking scores · cIFP table
```

### Tier Logic

| Analogs | Features |
|---------|---------|
| ≤ 20 | Full docking + pKaNET protonation + SMILES export |
| 21 – 200 | pKaNET protonation + SMILES export |
| > 200 | SMILES export only |

---

## How to Use

### Ligand-based (quickest)

1. Go to the app → select **Ligand-based**
2. Enter SMILES or draw in Ketcher → click **Load compound & continue →**
3. Click attachment atoms on the 2D structure → **Next**
4. *(Optional)* Enter binding-site residue names (e.g. `ASP315 PHE82 LEU83`) → ML ranks fragments instantly
5. Adjust number of analogs → **Generate**
6. Download SMILES CSV

### Structure-based — Mode A (have co-crystal PDB)

1. Select **Structure-based** → load PDB (search/ID/upload)
2. App detects ligand → confirms ✅ "Co-crystal ligand detected"
3. Choose **"Use this complex as-is"** → SMILES extracted automatically
4. Click attachment atoms → **Next**
5. Step 3: click **Run PLIP analysis** → see interaction table + recommendations
6. Click **Analyse unoccupied space** → 3D sub-pocket viewer appears
7. Select sub-pocket to filter fragment size → **Generate**
8. Step 5: **Run PLIP / cIFP on all poses** → compare parent vs analogs

### Structure-based — Mode B (have apo PDB)

1. Select **Structure-based** → load protein-only PDB
2. App detects apo → shows ℹ️ "Apo structure detected — docking will run automatically"
3. Enter parent SMILES → click attachment atoms → **Next**
4. Step 3: click **Dock ligand now** (ACD runs) → docked pose builds pseudo-complex
5. Continue as Mode A from here

---

## Installation (Streamlit Cloud)

The app is deployed at [https://analogbuilder.streamlit.app](https://analogbuilder.streamlit.app) — no local setup required.

To run locally:

```bash
git clone https://github.com/nyelidl/AnalogBuilder
cd AnalogBuilder
pip install -r requirements.txt
streamlit run app.py
```

### Requirements summary

```
streamlit==1.58.0          # web framework
rdkit==2025.9.6            # cheminformatics
anyonecandock==1.1.3       # docking engine (AutoDock Vina 1.2.7 auto-downloaded)
gemmi==0.7.5               # PDB/mmCIF parsing
plip==2.3.0                # interaction analysis (uses system obabel)
py3Dmol>=2.0.1             # 3D visualisation
transformers>=4.35,<4.45   # ChemBERTa (optional — falls back to rule-based)
torch>=2.0                 # ChemBERTa inference (optional)
scipy>=1.10                # statistics
```

**`packages.txt`** (system binaries via apt):
```
openbabel   # required by PLIP and ACD
```

---

## Project Structure

```
AnalogBuilder/
├── app.py                  # Streamlit UI (3,076 lines)
├── core.py                 # Chemistry backend + fragment library (6,188 lines)
├── pocket_reference.py     # ChemBERTa ML ranker (400 lines)
├── void_analyzer.py        # Pocket void geometry (461 lines)
├── plip_analyzer.py        # PLIP interaction analysis (472 lines)
├── requirements.txt
├── packages.txt
└── .streamlit/
    └── config.toml         # Disables file watcher (prevents torchvision warnings)
```

---

## Known Limitations

- **ChemBERTa is not fine-tuned** for drug–pocket interaction. Scoring is zero-shot.
- **No predicted ΔIC₅₀** — a supervised potency predictor is planned but not yet implemented.
- **Fragment library is small** (76 entries) — covers common building blocks only.
- **RMSD vs co-crystal** requires Mode A (co-crystal reference). Not available in Mode B.
- **Python 3.14 compatibility** — `tokenizers` cannot compile on Py3.14 (Streamlit Cloud 2026); ChemBERTa falls back to rule-based scoring silently.
- Docking scores are screening-level estimates, not free energy predictions.

---

## References

1. Chithrananda S, Grand G, Ramsundar B. ChemBERTa: Large-scale self-supervised pretraining for molecular property prediction. *arXiv* 2020, 2010.09885.
2. Salentin S et al. PLIP: fully automated protein–ligand interaction profiler. *Nucleic Acids Res* 2015, 43, W443–W447.
3. Eberhardt J et al. AutoDock Vina 1.2.0. *J Chem Inf Model* 2021, 61, 3891–3898.
4. Radoux CJ et al. Identifying interactions that determine fragment binding at protein hotspots. *J Med Chem* 2016, 59, 4314–4325.
5. Schmidtke P, Barril X. Understanding and predicting druggability. *J Med Chem* 2010, 53, 5858–5867.

---

## Suggested Citation

If you use Analog Designer in your research, please cite:

> Hengphasatporn K et al. Analog Designer: A Streamlit-Based Platform for ML-Guided Drug Analog Generation with Integrated Pocket Analysis, Docking, and Interaction Fingerprinting. *J Chem Inf Model* (in preparation).

And acknowledge the underlying tools: RDKit, AnyonCanDock, AutoDock Vina, PLIP, ChemBERTa, py3Dmol, OpenBabel.

---

## License

MIT License — see `LICENSE` for details.

---

## Project Status

Under active development. Interfaces and features may change. Feedback welcome via [GitHub Issues](https://github.com/nyelidl/AnalogBuilder/issues).
