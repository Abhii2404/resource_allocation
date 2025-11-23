# Cloud Resource Allocation — N2TC + GATA

A small research/demo repository that implements a hybrid cloud scheduling approach: N2TC (neural task classifier) + GATA (genetic algorithm) and compares it to several baseline schedulers. The project includes scripts to run the simulation, visualize results as PNGs, and a Streamlit dashboard to inspect saved results.

---

## Contents

- `main.py` — Main driver: loads data, trains N2TC, runs GATA, runs baseline comparisons, saves metrics and generates plots.
- `n2tc_module.py` — Implements the N2TC classifier (MLP), label generation and training/prediction helpers.
- `gata_module.py` — Genetic algorithm (GATA) used to create resource assignments and fitness evaluation.
- `baselines.py` — Several baseline scheduler simulations (FIFO, SJF, Mezmaz, Mocanu proxy) returning metrics.
- `visualizer.py` — Matplotlib-based routines to save comparison figures as PNG files (Fig5–Fig8).
- `app.py` — Streamlit dashboard to view `results_data.json` (relative performance and per-task trajectories).
- `data_loader.py` — Data loading and preprocessing helpers (used by `main.py`).
- `requirements.txt` — Python dependencies (install using pip).
- `results_data.json` — (generated) output saved by `main.py` containing metrics and extra info.

---

## Quick Start (Windows PowerShell)

1. Create / activate the Python environment you want to use (recommended: 3.10+).

```powershell
# Create a venv
python -m venv .venv
# Activate
.\.venv\Scripts\Activate.ps1
# Upgrade pip
python -m pip install --upgrade pip
# Install dependencies
python -m pip install -r requirements.txt
```

2. Run the main simulation (this trains N2TC, runs GATA, compares baselines, saves `results_data.json`, and generates plots):

```powershell
python main.py
```

3. Start the Streamlit dashboard (uses `results_data.json` saved by `main.py`):

```powershell
python -m streamlit run app.py
```

Open the URL printed by Streamlit in your browser (usually `http://localhost:8501`).

---

## Quick Start (WSL / Linux / macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
python -m streamlit run app.py
```

---

## Notes on Output

- `results_data.json` contains the simulation metrics and an `extra` object with the relative performance scores and a small ASCII table used by the dashboard.
- Figures saved by `visualizer.py` are:
  - `Fig5_Execution_Time.png`
  - `Fig6_Utilization.png`
  - `Fig7_Costs.png`
  - `Fig8_Response_Time.png`

---

## How Scores Are Computed

- The N2TC `Training Score` printed during run is the sklearn classifier accuracy on the held-out test split.
- The scheduling algorithms (baseline and proposed) are summarized with a composite, normalized "relative performance" score (0..1) computed from averaged utility / response / cost / time metrics and saved to `results_data.json` for display in the Streamlit dashboard.

If you need an alternate metric (per-task cost-based accuracy or pairwise RMS differences), the repository includes helper logic that can be adapted easily in `main.py`.

---

## Development Notes

- If you make changes, run the small test flow:
  - `python main.py` (generate metrics)
  - `python -m streamlit run app.py`
- To push a new branch to remote (example):

```powershell
git checkout -b my-feature
git add -A
git commit -m "my changes"
git push -u origin my-feature
```

- If `git push` is rejected because your local branch is behind remote, run:
  - `git fetch origin`
  - `git pull --rebase origin <branch>` (resolve conflicts if any)
  - then `git push origin <branch>`

---

## License & Contribution

This repository is provided as-is for demonstration and research. Feel free to open issues or submit PRs. If you want a specific license added, tell me and I can add a `LICENSE` file.

---

If you want, I can also:
- Add a `Makefile` or `tasks.ps1` for common commands,
- Add a minimal test that runs `main.py` with synthetic data, or
- Generate a compact example `results_data.json` for testing the dashboard.

Which of those would you like next?