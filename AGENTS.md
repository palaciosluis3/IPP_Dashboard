# Project Overview
IPP Dashboard is an AI-powered public policy evaluation tool based on the **Policy Priority Inference (PPI)** model. It enables policymakers to prioritize policies and optimize public spending to achieve development goals (such as SDGs). 

# Architecture & Stack
This project operates on a locally orchestrated two-tier architecture:
- **Frontend (`app.py`):** Developed strictly with Streamlit. It manages a 5-step guided pipeline and the user session state.
- **Backend (`backend/` directory):** A sequential mathematical pipeline running Python scripts via `subprocess`. It relies on the `policy-priority-inference` package.
- **Outputs (`Outputs/`):** Dynamic directory where the system saves generated plots, the consolidated Excel report (`final_report_IPP.xlsx`), and the executive PDF summary (`Resumen_Recomendaciones_IPP.pdf`).

# Setup & Execution Commands
- **Environment Management:** This project uses an automated Python virtual environment (`.venv`).
- **Initial Setup:** Run `setup.bat` to create/repair the `.venv` and install `requirements.txt` via `pip`.
- **Launch Application:** Use `start_app.bat`. It will check for the `.venv` existence and launch streamlit using the local interpreter.
- **Manual Launch:** `%VENV_PATH%\Scripts\python.exe -m streamlit run app.py`

# Code Style & Conventions
- **Language:** Python (>=1.21, <2.0 for NumPy compatibility).
- **UI Styling:** The Streamlit interface uses a custom color palette (e.g., deep_blue, sky_blue). This is injected via CSS using `st.markdown(..., unsafe_allow_html=True)` inside `app.py`. Any new UI component must respect and utilize these predefined CSS rules.

# Critical AI Agent Rules & Constraints
The following rules are strict and must not be violated when refactoring or adding features:

1. **Time/Year Synchronization:** When modifying temporal data logic, you MUST ensure structural consistency across four specific template locations. The years must match perfectly in:
   - Year columns in `Templates/raw_indicators.xlsx`.
   - Year columns in the "Presupuesto" sheet of `Templates/raw_expenditure.xlsx`.
   - Year lists in the "Población" sheet of `Templates/raw_expenditure.xlsx`.
   - Year lists in the "IPC" sheet of `Templates/raw_expenditure.xlsx`.
   
2. **Dynamic Configuration Parsing:** The `app.py` frontend dynamically passes user inputs to the backend scripts using a regex-based search-and-replace function (`update_script_config`). **Never rename global variables** in the backend scripts (e.g., `QM_VALUE`, `RL_VALUE`, `threshold`, `YEARS_TO_FORECAST`). Changing these variable names will instantly break the UI-Backend integration.

3. **Strict Execution Order:** The backend scripts must always be executed in the exact following sequence:
   1. `indicators_preparation.py`
   2. `interdependency_networks.py`
   3. `expenditure_preparation.py`
   4. `model_calibration.py`
   5. `prospective_simulation.py`
   6. `prospective_simulation_increase.py`
   7. `final_report_generator.py`
   8. `prospective_simulation_byconsideration.py`

4. **File Path Management:** All file read/write operations must exclusively use the `get_path` helper function defined in `app.py`. This ensures relative file paths resolve correctly regardless of the directory from which the launcher is executed.

5. **Goals vs. Real Goals (`goals` / `real_goals`):** `indicators_preparation.py` produces two target columns. `goals` is the simulation input (`G=goals` in `run_ppi`): it equals the real target when that target is still above the last level (`real_goal > IF`), otherwise it is a minimally *inflated* target `min(IF * GOAL_INFLATION_FACTOR, 1 - EPS)` (default `GOAL_INFLATION_FACTOR = 1.01`, i.e. 1% above `IF`, capped to stay inside the open interval `(0,1)`). This inflation only exists to avoid the PPI error that occurs when a goal is already reached at the last observation. `real_goals` is the true (normalized) government target. The same `GOAL_INFLATION_FACTOR` is also reused for the static-indicator nudge (`IF *= GOAL_INFLATION_FACTOR` when `I0 == IF`), since both cases only need to break an equality with a minimal push. **All plotting and reporting must use `real_goals`** — convergence donuts in `prospective_simulation.py` / `prospective_simulation_increase.py`, and the compliance/recommendation logic in `final_report_generator.py`. Never use the inflated `goals` for visuals or the final table.

6. **Full set saved, filtered at plot time:** `prospective_simulation.py` and `prospective_simulation_increase.py` save the **complete, unfiltered** indicator set to `output_baseline.xlsx` / `output_increase.xlsx`. The SDG filter (from `Outputs/selected_sdgs.json`) is applied **in memory only**, for plotting. This lets the "graphics-only" mode re-filter to any SDG selection without re-running calibration/simulation.

7. **Interpolation is for the interdependency network ONLY (`indicators_preparation.py`):** The model's core inputs (`I0`, `IF`, `successRates`) are computed **exclusively from real observations** — `I0` = first valid value, `IF` = last valid value, never interpolated/extrapolated. The interpolated **year columns** in `data_indicators.xlsx` are consumed **only** by `interdependency_networks.py` (which needs complete series for correlations); `model_calibration.py` and the simulations read the `I0`/`IF`/`successRates`/`goals` columns instead. Gap-filling uses **linear interpolation** for internal gaps and **linear extrapolation** (slope of the nearest segment, *not* flat replication of the nearest value) for missing leading/trailing years, hard-clipped to `[0, 1]`. Required by IPP and enforced in the script:
   - **Consecutive years**: year columns must be uniformly spaced by 1 (raises otherwise).
   - **Minimum observations**: each indicator needs at least `ceil(n_years / 2)` real observations (also rejects fully-empty rows); halts with a per-indicator report otherwise.
   - **Static indicators (`I0 == IF`) are allowed** — PPI accepts them but calibration needs `I0 != IF`, so `IF` is set to `IF * 1.05` (capped to the midpoint toward 1 if that would exceed the hard bound). This does **not** halt the pipeline.

# Graphics-Only Mode
The home page (Step 1) offers two paths via `st.session_state.mode`:
- **`'full'`** — the standard 6-step pipeline (upload → governance → params → SDGs → run → results).
- **`'graphics'`** — shortcut shown only when previous results exist (`output_baseline.xlsx` + `output_increase.xlsx`). It jumps straight to SDG selection (Step 4) and then runs **only** `backend/graphics_only.py`, which regenerates every plot, the final table, the PDF/MD summaries, and the by-consideration charts by reusing the saved `Outputs` (no recalibration/simulation). `graphics_only.py` reuses `final_report_generator.generate_report()` and `prospective_simulation_byconsideration.generate_plots_by_consideration()`, and replicates the baseline/increase plotting locally. Its `YEARS_TO_FORECAST` / `INTERMEDIATE_CONVERGENCE_YEAR` constants are kept in sync by `app.py` (Step 3), same as the other scripts.

Before running graphics generation, the app shows an `@st.dialog` confirmation modal (Cancel / Confirm) reminding the user to close any open result files in `Outputs`, since Windows file locks (e.g., an open PDF) cause "Permission denied" write errors.

# Relationship to the Original IPP Notebooks (`IPP_Código_Original/`)
The `backend/` scripts derive from the original PPI notebooks by Guerrero & Castañeda (the 5 `.ipynb` files in `IPP_Código_Original/`). The original notebooks required **many manual data transformations** before they could run; the dashboard's purpose is to **automate those manual steps**. A line-by-line review confirmed the **core PPI model is preserved unchanged**: the `ppi.calibrate` / `ppi.run_ppi` calls and their arguments, the 1000-run Monte Carlo, the adjacency-matrix construction, the lagged-correlation network (threshold `0.5`), the prospective `Bs` (tile of the last period), and the linear budget-growth ramp are all identical to the originals.

The following are **intentional, validated deviations** added to automate manual preprocessing and ease IPP implementation — they are correct and deliberate, not accidental regressions:

1. **`indicators_preparation.py` has no original counterpart.** The original notebooks already start from a prepared `data_indicators.xlsx`. All normalization, `I0`/`IF`, `successRates`, `goals`/`real_goals`, and interpolation/extrapolation logic is original to this app (following standard PPI conventions). See rules 5–7 above.
2. **`interdependency_networks.py` adds two heuristic refinements** after the `0.5` correlation threshold (not in the original): (a) removing negative intra-SDG edges (using `color` as the SDG proxy), and (b) resolving contradictory bidirectional loops (keeping the higher-magnitude edge). These shape the network matrix `A` and are deliberate.
3. **`expenditure_preparation.py` transforms the budget before extending it** (not in the original): (a) conversion to **real, per-capita terms** (`value / (population × CPI)`, hence the Población/IPC sheets in `raw_expenditure.xlsx`), and (b) **linear detrending** (`y − trend + mean`). These shape the budget matrix `Bs` and are deliberate.
4. **Convergence donuts use `real_goals`** (the original used the inflated `goal`); the **budget "jumpers"** highlight in the increase donut is an added visual. Both are intentional (see rule 5).
5. **Parameterization & robustness:** fixed notebook constants are exposed as app-configurable variables (`T = 16*5` → `YEARS_TO_FORECAST × calibration_index`; `Growth = 3` → `BUDGET_GROWTH_FACTOR`; `threshold = 0.95` → user slider). The adaptive `calibration_index = round(50 / n_years)` also fixes a latent original bug where a non-10-year horizon desynchronized the `Bs` column count. Plus defensive validations, `multiprocessing`, plot grouping, `get_path`, and SDG filtering — all infrastructure/cosmetic.