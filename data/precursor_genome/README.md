# Precursor Genome

A dataset of 1,035 pairwise solid-state (ceramic) synthesis experiments performed by the
[A-Lab](https://newscenter.lbl.gov/2023/11/22/a-lab-at-berkeley-lab/) self-driving laboratory
at Lawrence Berkeley National Laboratory. Each experiment combines two inorganic precursor
materials, heats them together, and characterizes the resulting phase composition by X-ray
diffraction (XRD). Every sample's synthesis conditions, precursor masses, instrument settings,
Rietveld refinement results (phase weights, Rwp), and human-verified reaction category and
quality score are recorded in a single structured JSON ledger, validated against a Pydantic
schema.

## Contents

This repository holds everything needed to load and query the dataset. Large binary files
(raw diffraction patterns, refinement objects, plot images) are archived separately on
Zenodo — see [GitHub / Zenodo split](#github--zenodo-split) below.

| Path | Description |
|---|---|
| `ledger_precursor_genome.json` | The dataset itself — one JSON ledger, 1,035 samples (~30 MB) |
| `schemas_precursor_genome.py` | Pydantic schema defining the ledger's structure (`ExperimentRepositoryLedger` and its nested models) |
| `data_access.py` | Small helper used by notebook 03 to detect whether `data/` is present locally and print an informative message (with a Zenodo pointer) instead of crashing when it isn't |
| `notebooks/` | Three Jupyter notebooks demonstrating how to load, filter, and (optionally) plot the data — see [Notebooks](#notebooks) below |
| `outputs/` | Empty by default — notebook 02 writes filtered CSV exports here when run |
| `metadata/datacite.json` | DataCite-format metadata (authors, ORCIDs, license, resource type) for archival/indexing |
| `requirements.txt` | Python package requirements for the notebooks |
| `LICENSE` | CC BY 4.0 |
| `data/` | Raw scans, refinement objects, and plots — **archived on Zenodo, not included in this GitHub repository** (see below) |

## Dataset Summary

Computed directly from `ledger_precursor_genome.json` via
`schemas_precursor_genome.ExperimentRepositoryLedger.model_validate()`:

- **Samples (reactions):** 1,035
- **Unique precursors:** 46
- **Unique elements represented:** 39
- **XRD scans:** 1,351
- **Refinement cases:** 1,950
- **Reaction category breakdown:** transformed (384) · unreacted (369) · completely reacted (143) · partially reacted (113) · unclassified (26)

Each `SampleEntry` carries `sample_id`, `precursors` (formula, CAS number, purity), `target_compound`,
`target_stoichiometry`, `reaction_energy_ev_per_atom`, synthesis conditions (heating temperature/time,
furnace, precursor masses), the reaction `outcome` (category + phase composition), and XRD
characterization (scans and their Rietveld refinement cases — phase weights, Rwp, human quality
score).

## GitHub / Zenodo split

This GitHub repository contains everything **except** `data/` — the ledger, schema, notebooks,
metadata, and documentation. `data/` (raw `.xrdml` diffraction files, refinement `.pkl` objects,
and rendered `.png`/`.html` plots — roughly 17 GB) is archived separately on Zenodo, because it's
too large to host on GitHub.

**Zenodo archive DOI: [10.5281/zenodo.21285545](https://doi.org/10.5281/zenodo.21285545)** (also recorded in
`metadata/datacite.json` and `data_access.py`; the deposit currently has restricted access).

**The ledger alone, available in this GitHub repository, already contains full per-reaction
metadata with zero degradation** — thermal profiles, precursor/recovered masses, instrument
configuration, phase weights, Rwp scores, human quality scores, and reaction category
assignments are all stored directly as JSON fields, not in external files. You do not need
`data/` to load, filter, or analyze any of that.

Only operations that open an actual file under `data/` — plotting a raw XRD pattern, loading a
refinement `.pkl`, or displaying a rendered plot image — require the Zenodo download. Notebook
`03_plot_xrd_patterns.ipynb` is the only notebook that touches `data/`; if it isn't present
locally, the affected cells print an informative message (pointing to the Zenodo DOI above)
instead of crashing, while every ledger-native operation continues to work normally. See
`data_access.py` for the shared helper this relies on.

## Schema structure

Defined in `schemas_precursor_genome.py` (Pydantic v2). Top-level container:

```
ExperimentRepositoryLedger
├── repository_name, sample_dir, xrd_settings, workflow_spec
└── samples: List[SampleEntry]

SampleEntry                          one per synthesis experiment ("PG_XXXX")
├── sample_id, target_compound, target_stoichiometry, reaction_energy_ev_per_atom
├── precursors: List[Precursor]      formula, CAS number, melting point, purity
├── synthesis: SynthesisData         heating temperature/time, furnace, precursor masses
├── physical_failure: PhysicalFailure (optional)
├── outcome: OutcomeEntry            reaction_category, phases (name + weight_percent)
└── characterization: CharacterizationEntry
    ├── xrd: XRDCharacterization
    │   ├── scans: List[ScanEntry]
    │   │   ├── filename, scan_index, temperature, status, prep_verification
    │   │   └── refinement_cases: List[RefinementCase]
    │   │       ├── rank, rwp, phase_weights, cif_names
    │   │       ├── pkl_path, html_path, image_path   (files under data/, see above)
    │   │       └── verification: AnalysisVerification  (human quality score, acceptance)
    │   └── outcome: OutcomeEntry     (XRD-level phase snapshot)
    ├── sem: SemCharacterization      (precursor-level SEM images)
    └── eds: EdsCharacterization      (precursor-level EDS spectra/quantification)
```

`SampleEntry.best_refinement` is a convenience property that walks `active_scan_index` →
`active_case_index` to return the specific `RefinementCase` that is the sample's actual
reported result (as opposed to a superseded rerun or rejected candidate).

## Notebooks

| Notebook | Demonstrates | Needs `data/`? |
|---|---|---|
| `01_load_and_explore.ipynb` | Load the ledger with the Pydantic schema, build a flat per-sample DataFrame, compute summary statistics, inspect a single sample and its scans | No — ledger only |
| `02_filter_and_query.ipynb` | Filter samples by reaction category and by elements present in precursors, extract phase weights from each sample's reported refinement result, export filtered subsets to CSV | No — ledger only |
| `03_plot_xrd_patterns.ipynb` | Parse and plot raw `.xrdml` diffraction patterns; overlay multiple scans; load a refinement result `.pkl` and display its fit plot and rendered image | **Yes** — degrades gracefully with an informative message if `data/` isn't present locally |

## Citation

This dataset accompanies a manuscript in preparation. Authors and affiliations (see
[`metadata/datacite.json`](metadata/datacite.json) for the full DataCite record):

Lauren Walters, Matthew J. McDermott, Bernardus Rendy, Yuxing Fei, Gerbrand Ceder —
Lawrence Berkeley National Laboratory.

DOI: [10.5281/zenodo.21285545](https://doi.org/10.5281/zenodo.21285545)

Note on provenance: this dataset was produced using an internal Rietveld-refinement pipeline
built on non-public research software; that production tooling is intentionally not included
in this release, which is access-only.

## License

[CC BY 4.0](LICENSE) — you are free to share and adapt this material with attribution.
