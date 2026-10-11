# The target output — LOCKED 2026-10-06

This is the definition of done for the IRIS project (deadline 2026-10-13).
Every task should move one field below from missing or weak to present and
measured. A task that improves no field is a detour.

## The pipeline

```
INPUT:  a generated structure (CIF) from MatterGen, GNoME, or our DAO-G
        |
        v
  target composition, e.g. KRhO2
        |
   +----+-----------------------------------------+
   |                                              |
 DAO-S                                       DAO-P (ORB-v3)
 precursors + firing temperature             relax the structure, price it
   |                                              |
   +----------------+-----------------------------+
                    v
            reaction.assess()   <- the simulator
                    v
            FINAL OUTPUT: lab protocol + verdict
```

## The output, as the code produces it today

Computed by `synflow.protocol.build` for KRhO2, not written by hand:

```
TARGET   KRhO2   (batch 2 g)
REACTION 0.5 K2CO3 + 0.5 Rh2O3 -> KRhO2 + 0.5 CO2

1. WEIGH
     K2CO3             0.7943 g   (0.5 x 138.21 g/mol)
     Rh2O3             1.4587 g   (0.5 x 253.81 g/mol)

2. GRIND AND MIX
     agate mortar, about 45 minutes, a few drops of acetone

3. FIRE
     772 C, ramp 5 C/min, 2 cycle(s)
     atmosphere: air
       because no oxygen is exchanged

4. REGRIND AND REFIRE
     1 further cycle at the same temperature

FEASIBILITY  thermodynamics does not block this
     dH +1.284 eV   gas -2.316 eV   dG -1.031 eV
     thermodynamic floor about 307 C
```

Mass balance, checkable by hand: 0.7943 + 1.4587 = 2.2530 g in, 2.0000 g of
KRhO2 out. The 0.2530 g difference is the CO2 that leaves:
0.5 x 0.011494 mol x 44.01 g/mol = 0.2529 g.

## Every field, where it comes from, and its state

Three kinds of field, kept apart on purpose. Only PREDICTED fields carry a
measured error; DERIVED ones are arithmetic that is either right or wrong;
CONVENTION ones are standard practice that no corpus records.

| # | field | kind | source | state |
|---|---|---|---|---|
| in | CIF, directory, .zip, .extxyz | — | `scripts/assess_generated.py` | **done** |
| — | relax + energy of the target | predicted | `dao_g.relax` + ORB-v3 | **done**; cell held fixed — fine for generated input, wrong for template substitution |
| 1 | precursors | predicted | `ConditionalLookup` (metal, anion class, oxidation state) | **improved** 2026-10-07: 0.724 Jaccard, 55.7% exact on unseen (rule was 0.662 / 47.5%) |
| 1 | grams of each | derived | `protocol.weigh_out` | **done, wired** (2026-10-06); mass balance tested |
| 1 | purity grade | convention | `protocol.PURITY` | **done** 2026-10-07: 99.9%, dry hygroscopic precursors first; labelled a convention |
| 2 | balanced equation | derived | `ComputedReaction` | **done** |
| 3 | grinding | convention | `protocol` | **done, wired** |
| 4 | firing temperature | predicted | `TemperatureNet`, retrained on the lookup's precursors | **works**: 194 C MAE, 120 C median on unseen; wired via `dao_s.load_shipped` |
| 4 | dwell time | predicted | dwell network (`scripts/train_dwell.py`) | **retrained** 2026-10-08 on the fixed parser: within 2x for 48.6% (median 6 h baseline 45.7%), median 2.06x off (baseline 3.00x); rounded to conventional values |
| 4 | ramp rate | convention | `protocol` | **done, wired** |
| 5 | atmosphere | derived | which side O2 is on | **done, wired**; all three cases tested |
| 6 | firing cycles | derived/convention | corpus operations, else 2 | **done, wired** |
| V | dH, gas term, dG, floor / ceiling | derived | `reaction.assess` | **fixed** 2026-10-08: gases from the balanced equation (signed), 1 bar and in-air verdicts, one enthalpy source per reaction; A-Lab Table 2 formed 68/71 = 95.8% (was 88.7%) |
| V | kinetic-risk flag | derived | dG below 50 meV/atom of reactants | **done** (2026-10-06); always shown, so "checked" is visible |
| V | kinetic evidence per precursor pair | evidence | `kinetics.check_recipe` (Precursor Genome, 1,035 robot reactions) | **new** 2026-10-08: A-Lab Table 2 inert 7/10 (was 5/10), formed unchanged |

## The atmosphere rule, stated once so it cannot drift

| O2 in the balanced equation | atmosphere | example |
|---|---|---|
| consumed (reactant side) | air or flowing O2 | Co3O4 -> LiCoO2: Co +2.67 to +3 |
| released (product side) | **inert** (N2/Ar), reducing if needed | Fe2O3 -> LiFePO4: Fe +3 to +2 |
| absent | air, doing no chemistry | K2CO3 + Rh2O3 -> KRhO2 |

The first mock of this output said "air (O2 on the product side)". That is
self-contradictory: O2 on the product side means the target is more reduced
than its precursors and air would oxidise it back. The code had it right; the
mock did not.

## What "done" means for IRIS

1. This output produced end to end from a CIF, every field present or
   explicitly marked as not predicted.
2. Each predicted field measured on data the model never saw.
3. The simulator measured against real experimental outcomes — A-Lab's
   Supplementary Table 2, about 90 robotic reactions including "no reaction"
   cases, which test the gate rather than only confirm it.

## State as of 2026-10-08

Every field of the output now exists. The three kinds are kept apart, and only
the predicted ones carry an error bar:

| predicted field | unseen / held-out | baseline | oracle ceiling |
|---|---|---|---|
| precursors, per metal (validation) | 80.9% lookup, 82.7% network | 65.6% plain rule | **92.9%** |
| precursors, set (unseen, 981 targets) | 0.724 J, 55.7% exact | 0.662 J, 47.5% | -- |
| firing temperature (test) | 167.0 C MAE, 116.8 C median | 258.7 C | **79.2 C** (119.8 C on multi-recipe targets) |
| dwell time (test) | 48.6% within 2x, 2.06x median | 45.7%, 3.00x | **80.7%** |

### Against a real autonomous lab (A-Lab, Szymanski et al. 2023)

The simulator, Supplementary Table 2 (`scripts/eval_alab.py`, 1 bar):

| | before 2026-10-08 | now |
|---|---|---|
| formed (robot made it; gate should not reject) | 63/71 = 88.7% | **68/71 = 95.8%** |
| inert (robot saw nothing; gate should not expect a product) | 5/10 = 50% | 5/10 thermo only; **7/10 with kinetic evidence** |

What changed: gases are read from the balanced equation (oxygen released by
redox was never counted, and 2 BaCO3 counted as one CO2 -- five of the eight
rejections), a target written unreduced (Mn2P4O12) now balances, and the
kinetic evidence of the Precursor Genome vetoes a pair the robots saw stay
unreacted at a higher temperature. The three remaining rejections are all
gas-free oxide + oxide reactions (YFeO3 +0.04 eV, within error; ZnNiO2, a
solid solution with no mixing entropy in the model; CuTa2O6).

The kinetic gates that were tried and NOT kept, measured the same way: a
logistic model trained on the Precursor Genome (formed 95.8% -> 74.6%; its
temperatures are tied to its precursors by design) and Tammann's rule as a
veto (inert 80% but formed 93.0%, and no power inside the Precursor Genome).

The whole system, the robot's 57 targets (`scripts/eval_alab_targets.py`;
54 not in our training corpus):

| field | ours vs the robot's recipe (40 unseen targets with a recipe) |
|---|---|
| precursors | 42.5% exact set, Jaccard 0.664, 76.7% per metal -- never a wrong metal; every miss is another source of the same metal (8 of 32: NH4H2PO4 for (NH4)2HPO4) |
| temperature | 166 C MAE, 135 C median; we fire +107 C hotter on average, and the robot never went above 1100 C |
| dwell | 57.5% within 2x of its 4 h |
| grams | 42/42 identical to what the robot weighed (largest mass-fraction gap 0.0003) |
| simulator vs Success/Failed | running |

**Does the network beat a lookup?** (`scripts/knn_baseline.py`, k chosen on
validation, both scored on the same test rows, precursors from the lookup as
shipped)

| | network | k-nearest neighbours | average of the two |
|---|---|---|---|
| temperature, n = 4,713 | MAE 175.1 C, median 122.6 C | MAE 175.3 C, **median 100.0 C** | MAE 167.9 C |
| dwell, n = 2,652 | **54.1% within 2x** | 53.2% | 50.6% |

| unseen, 981 targets | MAE 194 C, median 120 C | MAE 193 C, **median 100 C** | MAE 188 C, median 116 C |

On temperature the 445,185-parameter network does not beat looking up the 25
most similar compounds -- on test, on unseen, or in the median; on dwell it is
a point ahead. The 167.0 C quoted above was the network fed the corpus's own
precursors; as shipped, with the lookup's, it is 175.1 C on these rows.

The average was NOT shipped. The rule, stated before computing it: ship only
if it lowers validation MAE by at least 5 C. It lowered it by 4.9 C (190.4 ->
185.5) and made dwell worse (54.6% -> 50.9%). Its better test and unseen
numbers were seen after, or alongside, that decision and do not overturn it.

**The corpus check** (`scripts/validate_simulator.py`, 40 recipes from the
test split): 15 of 22 checkable real recipes accepted; 18 unchecked, mostly
targets Materials Project lacks. Its scrambled negative set turned out to
test nothing -- see the script -- so the gate's negatives are the robots'.

The oracle is told each held-out compound's own recorded answers and picks the
best single one, so no model can beat it. It answers "why not 95%": the data
caps per-metal precursor accuracy at 92.9%, because one compound's metal is
supplied by different precursors in different papers. It also shows the room
that is left -- about 10 points on precursors, 50-90 C on temperature, and 30
points on dwell time, where the model barely moved from its baseline.

**Training longer is not the lever.** In every run the training loss keeps
falling while validation stops improving -- temperature at about epoch 25,
the precursor network at 18, dwell at 16 -- which is memorising, not learning.

### Queued, one heavy job at a time (the machine is an i3)

0. **Results of 2026-10-08 night, to carry into the paper:**
   * **MatterGen's four lab-tested candidates** (`scripts/eval_mattergen.py`,
     Nature 2025 SI D.10): the verdict on the LAB's recipe matched the lab for
     3 of 4 -- TaCr2O6 feasible (made), Cr2MoO6 and Mn3NiO6 uphill/uncertain
     (failed). LaMoO4 was called feasible and failed by Mo volatilisation,
     which nothing here models; Cr2MoO6 also lost Mo at 900 C.
   * **Novel targets get wide error bands**: +/-111 to +/-643 meV/atom (known
     targets +/-38), from hull's composition-linear calibration. So most
     generated materials come out "uncertain". Next fix: anchor a novel
     target to its MP decomposition phases with ORB on both (to be validated
     on known compounds first).
   * **Selectivity v3** (pair rivals only, plus remaining driving force): AUC
     0.459 (C1) and 0.497 (remaining) on A-Lab made vs failed. Three versions,
     no signal: kept as information, never as a verdict.
   * **ARROWS3 YBCO** (149 runs, judged as O7): pooled, yield follows
     temperature (rho +0.79) and no metric beats it; within one temperature
     dG ranks recipes at 800 C (rho -0.49) but not at 900 C (+0.20); C1 has
     the wrong sign at both. Rerun at the paper's O6.5 (Ba4Y2Cu6O13), all
     149 scored: pooled dG -0.149, C1 -0.176, remaining -0.151 against
     temperature's +0.786; within 800 C dG -0.42, within 900 C +0.09. The
     Colab "all unchecked" was the script resuming its own failed rows --
     fixed: unchecked rows are rescored, the file is keyed by the judged
     target.
   * **Volatility warning** (`synflow/volatility.py`, 2026-10-09): Mo, Sb,
     Bi, Pb, V, Li, Na, K in any precursor, above the melting point of the
     volatile oxide or carbonate (Precursor Genome table). Flags both of
     MatterGen's Mo-loss failures at 900 C and neither success. On the
     A-Lab's targets it flags 50% of the made and 33% of the failed, so it is
     a precaution (lid, excess, lower T), not a predictor of failure.

0b. **2026-10-09 afternoon:**
   * **Anchoring validated and switched on.** 41 A-Lab targets MP has, each
     treated as unknown: median |error| 0.0068 eV/atom anchored vs 0.0352
     calibrated; MAE 0.0109 vs 0.0480 on 40 (BaGdCrFeO6 wrong by ~3 eV under
     both); anchored closer on 30/40. Band uses the RMS, 0.022 eV/atom. The
     calibration's stated error (0.117) was 2.4x its real one (0.048).
   * **MatterGen, rerun with anchoring:** all four candidates sit 7-27
     meV/atom above their decomposition phases, so stability does not
     separate them. Lab recipes: TaCr2O6 (made) feasible; Cr2MoO6 uncertain
     (+/-56 meV/atom, kinetic risk); LaMoO4 and Mn3NiO6 feasible though
     failed. The volatility check flags both Mo losses at 900 C. Net: the
     success confirmed, two of three failures flagged, Mn3NiO6 missed --
     replacing the earlier "3 of 4", which rested on calibration bands wide
     enough to call Mn3NiO6 uphill by accident.
   * Verdicts now also state a novel target's height above the phases it
     would decompose into, flagged above 44 meV/atom (2x the anchor RMS).

1. **Selectivity** on the A-Lab's targets. v2 (C1 alone): AUC 0.486 --
   no separation of the 14 failures from the 42 made; 90% of the MADE ones
   had a more favoured rival too. Reading the rows showed single-precursor
   decompositions (CaCO3 -> CaO) counted as rivals. v3, running: rivals
   must react two precursors, and the remaining driving force from the
   rival to the target -- the A-Lab's own account of its stalls -- is added.
   Hypotheses written in `scripts/eval_selectivity.py` before the run.
2. **validate_simulator.py** -- running, 40 real + 40 scrambled recipes.
3. **Nearest-neighbour baseline** (`scripts/knn_baseline.py`).
4. **200-epoch temperature run** -- to settle "more epochs" with a curve.
   Prediction, stated first: validation flat or worse.

Fixed 2026-10-08 evening, after an audit of the whole path (`runs/e2e_demo_fixes2.log`):

* **A generated polymorph of a known formula was judged as the known phase.**
  With a structure given, its energy is now MP's known phase plus ORB's gap
  between the two structures (both relaxed, both scored by ORB: 0.0124 eV/atom
  MAE), and the gap is reported. BaTiO3 and LiCoO2 demo CIFs: +0 meV/atom
  against mp-5777 and mp-22526 -- the sanity check.
* **Every verdict now has an error band** -- 35 meV/atom for an all-MP reaction
  (Bartel et al. 2019), 2 sigma for calorimetry, plus the potential's
  calibration on a novel target -- and inside it the verdict is UNCERTAIN.
  KRhO2: dG -17 meV/atom inside +/-168, so uncertain, not "feasible".
  YFeO3 in Table 2 moves from rejected to uncertain (+8.7 meV/atom).
* **The predicted temperature is checked** against the floor, the ceiling and
  a box furnace's ~1200 C (BaTiO3 at 1234 C is flagged).
* A structure whose relaxation failed is no longer scored.
* **Label noise in the unseen set**: 33 of 981 "firing temperatures" are below
  300 C (NdSiAlON at 40 C); they carry 17% of the error. Without them the
  network's unseen MAE is 166 C, not 194 C -- a sensitivity, not a headline,
  since the 300 C line was drawn after looking.

Done 2026-10-08, the demo rerun (`runs/e2e_demo_after_fixes.log`):

| target | DAO-S | dG at 1 bar | dG in air | verdict |
|---|---|---|---|---|
| BaTiO3 | BaCO3 + TiO2, 1234 C, ~3 h | -2.09 eV (all measured) | -3.10 eV | feasible, floor 289 C |
| KRhO2 | K2CO3 + Rh2O3, 970 C, ~12 h | -0.09 eV | -0.51 eV | barely: floor 886 C (was 306 C before the gas fix), 17 meV/atom, KINETIC RISK |
| LiCoO2 | Co3O4 + Li2CO3, 884 C, ~16 h | -0.96 eV | -1.33 eV | feasible, 174 meV/atom |

### Known limits, for the write-up

* The simulator assumes the target is the ONLY solid product. A-Lab's
  Na2CO3 + FePO4 -> NaFeP2O7 cannot balance that way (P:Fe is 1:1 in FePO4
  and 2:1 in the product), so another iron phase must have formed too.
* Gas-free oxide + oxide routes are where both the simulator and the kinetic
  flag are weakest: no entropy term, and a driving force within the error of
  the energies.
* Measured (calorimetric) enthalpies are used only when EVERY species in a
  reaction has one. Mixed with Materials Project they fail -- per compound
  the two differ by up to 0.32 eV/atom (Ta2O5), and CuO + Ta2O5 -> CuTa2O6
  came out -2.15 eV mixed against +0.34 all-MP. A generated material is
  never measured, so its reaction is always all-MP.
* Kinetic evidence covers only pairs of the Precursor Genome's 46 precursors,
  and the inert test it was measured on has 10 rows.
* Five Table 2 rows stay unchecked: their products (CaGd3O6, AgV4O11) and one
  precursor (Y2(CO3)3) are in neither Materials Project nor the A-Lab's CIFs.
* The potential cannot replace Materials Project for precursor energies:
  median 0.2-0.3 eV off on 29 corpus reactions, worst on f-electron elements
  and the anions MP corrects empirically.
