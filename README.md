# SynFlow-Mat

**From a crystal file to a checked laboratory protocol.**

Generative models such as MatterGen and GNoME propose new inorganic crystals,
but they do not say how to make them. SynFlow-Mat takes a crystal structure
(a CIF file) and returns a complete solid-state synthesis protocol, then
checks that protocol before anyone switches on a furnace. It also says
"uncertain" when the energies cannot decide.

The protocol covers:

- precursors and their masses
- grinding
- firing temperature and hold time
- atmosphere
- number of firings

The check uses:

- the reaction's Gibbs free energy, counting the gases it releases or consumes, with an error band
- what autonomous-laboratory robots have already observed for the same precursor pairs
- warnings for volatile precursors and for furnace limits

Every part was tested against outcomes from autonomous laboratories that the
system never saw in training. Where the system does not work, this README
says so.

---

## Example

```
python scripts/assess_generated.py --input examples/BaTiO3.cif --show-protocol
```

```
TARGET   BaTiO3   (batch 2 g)
REACTION BaCO3 + TiO2 -> BaTiO3 + CO2

1. WEIGH
     BaCO3             1.6925 g   (1 x 197.34 g/mol)
     TiO2              0.6850 g   (1 x 79.87 g/mol)
2. GRIND AND MIX
     agate mortar, about 45 minutes, a few drops of acetone as a grinding aid
3. FIRE
     1234 C, ramp 5 C/min, 2 cycle(s)
     hold ~3 h per cycle (predicted; typically within a factor of 2)
     atmosphere: air
     CHECK: 1234 C is above a standard box furnace (~1200 C): needs MoSi2 elements or a tube furnace
4. REGRIND AND REFIRE
     1 further cycle(s) at the same temperature

FEASIBILITY  thermodynamics does not block this
     dH +1.171 eV   gas -3.340 eV   dG -2.169 eV   (gases at 1 bar)
                     gas -4.350 eV   dG -3.179 eV   (in air: CO2 4.2e-4 bar, O2 0.21 bar)
     thermodynamic floor about 255 C
     error band +/-38 meV/atom -- what these energies can resolve
     polymorph: +0 meV/atom against the known phase mp-5777
```

Each value in the protocol is labelled as one of three kinds:

- *predicted*: from a model trained on text-mined recipes
- *derived*: computed, for example the masses or the atmosphere
- *convention*: standard laboratory practice, for example the number of regrinds

---

## Install

```
git clone <this repository>
cd SynFlow-Mat-IRIS
pip install -r requirements.txt
```

Python 3.13, CPU only, is enough. Reaction energies come from the
[Materials Project](https://next-gen.materialsproject.org/), which needs a
free API key:

```
# Windows (PowerShell)
$env:MP_API_KEY = "your-key"
# Linux / macOS
export MP_API_KEY=your-key
```

Run the tests (about 30 seconds):

```
python -m pytest -q tests
```

---

## What is in this repository

| Path | What |
|---|---|
| `synflow/dao_s.py`, `recipe_metrics.py`, `precursor_net.py` | Recipe prediction: precursors (conditional lookup), firing temperature (neural network), hold time (neural network) |
| `synflow/protocol.py` | Turns a predicted recipe into the written protocol: masses, grinding, firing, atmosphere, regrinds, checks |
| `synflow/reaction.py`, `thermo_data.py` | Reaction balancing and free energy ΔG = ΔH − TΔS, with gas entropy, the error band and the "uncertain" verdict |
| `synflow/hull.py` | Energy of a novel target, anchored to its decomposition products |
| `synflow/kinetics.py` | Evidence from 1,035 robot-run pairwise reactions (Precursor Genome) |
| `synflow/volatility.py` | Warning for precursors that melt or evaporate below the firing temperature |
| `synflow/selectivity.py` | Selectivity metrics (competing reactions), tested and reported as giving no signal |
| `synflow/dao_g.py`, `dao_p.py`, `potentials.py` | Structure relaxation with the ORB interatomic potential |
| `scripts/` | Command-line entry point (`assess_generated.py`) and every evaluation in the paper |
| `runs/` | The trained models the protocol uses (about 4 MB) |
| `results/` | The exact outputs behind the numbers below |
| `data/benchmark/` | Test cases built for this project |
| `data/precursor_genome/` | Precursor Genome ledger (CC BY 4.0, see Data) |
| `examples/` | Three CIF files to try |
| `paper/figures/` | Figures used in the paper |
| `docs/target_output.md` | Specification of the protocol output |

---

## Results and how to reproduce them

All tests use experiments the models were not trained on.

| Test | Result | Script | Output in `results/` |
|---|---|---|---|
| A-Lab pairwise reactions (Table 2 of Szymanski et al. 2023) | The feasibility check agreed with **95.8 %** of the reactions the robot made (88.7 % before gases were taken from the balanced equation), and with **70 %** of those that did not react when kinetic evidence is used | `eval_alab.py --enthalpies measured` | `alab_table2_measured.json` |
| A-Lab's 57 targets: precursor masses | **42 / 42** recipes match the masses the robot weighed | `eval_alab_targets.py` | `alab_targets.json` |
| A-Lab's 57 targets: precursors | 76.7 % of per-metal precursor choices match | `eval_alab_targets.py` | `alab_targets.json` |
| A-Lab's 57 targets: temperature | MAE 166 °C; the system fires on average 107 °C hotter than the robot | `eval_alab_targets.py` | `alab_targets.json` |
| Unseen literature recipes (981 targets): temperature | MAE 194 °C (median 120 °C), against 279 °C for predicting the mean; a k-nearest-neighbour baseline scores the same (193 °C) | `eval_unseen.py`, `knn_baseline.py` | `unseen_evaluation*.json` |
| Unseen literature recipes: precursors | 55.7 % exactly right; Jaccard similarity 0.72 | `eval_unseen.py` | `unseen_evaluation.json` |
| Hold time | 48.6 % within a factor of 2 (median baseline 45.7 %) | `train_dwell.py` | `dwell_result.json` |
| Energy of novel targets: anchoring vs calibration, 41 known compounds treated as unknown | Median error 0.0068 vs 0.0352 eV/atom | `validate_anchoring.py` | `anchoring_validation.jsonl` |
| Which A-Lab targets fail? Three selectivity measures | AUC 0.486, 0.459, 0.497: **no signal** (0.5 = chance) | `eval_selectivity.py --alab` | `selectivity_v3.jsonl` |
| ARROWS3, YBa2Cu3O6.5, 47 precursor sets | Pooled Spearman ρ: ΔG −0.149, firing temperature +0.786. Temperature, not thermodynamics, explains the yield | `eval_arrows3.py --target Ba4Y2Cu6O13 --decompose-missing` | not included (see Data) |
| Four MatterGen candidates tried in a laboratory (one was made) | The made one (TaCr2O6) is judged feasible. Of the three failures: one is "uncertain", one is flagged volatile, and one (Mn3NiO6) is judged feasible, which is a miss | `eval_mattergen.py` | `mattergen_showcase.json` |

`scripts/make_paper_figures.py` redraws `paper/figures/` from these files.

### Limitations

These limits are stated plainly:

- **Failures cannot be predicted.** No thermodynamic measure tested here can tell which targets a laboratory will fail to make.
- **Temperature is limited by the data itself.** Different papers report different temperatures for the same compound. In the unseen test set, 33 of the 981 labels are below 300 °C, which is not a solid-state firing, and they carry 17 % of the error.
- **The error band is wide.** Reaction energies from Materials Project are accurate to about 35 meV/atom (Bartel et al. 2019), which is why some verdicts are "uncertain".
- **This is a first attempt, not a guarantee.** The system proposes a checked starting recipe. It has not been tested in a new laboratory experiment.

---

## Data

Some data is downloaded automatically; the rest you fetch yourself.

**Downloaded automatically from figshare** the first time it is needed, and checked by MD5:

- Kononova et al. 2019 text-mined recipes (figshare 9722159)
- The LLM-extracted corpus of 80,806 syntheses (figshare 30423274)
- Kingsbury et al. 2022 calorimetric enthalpies (figshare file 28232538)

**Included:** the Precursor Genome ledger (Walters, McDermott, Rendy, Fei and Ceder, [github.com/lauren-walters/precursor-genome](https://github.com/lauren-walters/precursor-genome), CC BY 4.0).

**Needed only to re-run the evaluations; download them yourself:**

| Data | Put it at | Source |
|---|---|---|
| A-Lab Supplementary Information (PDF), converted to text | `data/alab/alab_SI.txt` | Szymanski et al., *Nature* 624, 86 (2023), Supplementary Information |
| A-Lab supplementary data | `data/alab/alab_supplementary_data.zip` | same article, Supplementary Data (`41586_2023_6734_MOESM3_ESM.zip`) |
| MatterGen experimental CIFs | `data/mattergen/experimental/*.cif` | [microsoft/mattergen](https://github.com/microsoft/mattergen) `data-release/cifs.zip`, folder `cifs/mattergen/experimental/` |
| ARROWS3 YBCO experiments | `data/arrows3/YBCO_Exp.json` | [njszym/ARROWS](https://github.com/njszym/ARROWS) `Examples/YBCO/Exp.json` (no licence is stated, so it is not redistributed here) |

To convert the SI PDF to text:

```
python -c "import pypdf; r=pypdf.PdfReader('data/alab/alab_SI.pdf'); open('data/alab/alab_SI.txt','w',encoding='utf-8').write(''.join('\n<<<PAGE %d>>>\n' % (i+1) + (p.extract_text() or '') for i, p in enumerate(r.pages)))"
```

---

## Acknowledgement of AI assistance

The code was written with the help of an AI assistant (Claude, by Anthropic):

- **The author:** the project idea and goals, and the final decisions.
- **The AI assistant:** helped design, write and debug the code, and check the results.

Every number reported above comes from running the scripts in this repository.

## Licence

- **Code:** MIT (see `LICENSE`).
- **Precursor Genome data:** CC BY 4.0 (see `data/precursor_genome/LICENSE`).
