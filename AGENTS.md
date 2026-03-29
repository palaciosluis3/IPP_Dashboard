# Project Overview
IPP Dashboard is an AI-powered public policy evaluation tool based on the **Policy Priority Inference (PPI)** model. It enables policymakers to prioritize policies and optimize public spending to achieve development goals (such as SDGs). 

# Architecture & Stack
This project operates on a locally orchestrated two-tier architecture:
- **Frontend (`app.py`):** Developed strictly with Streamlit. It manages a 5-step guided pipeline and the user session state.
- **Backend (`backend/` directory):** A sequential mathematical pipeline running Python scripts via `subprocess`. It relies on the `policy-priority-inference` package.
- **Outputs (`Outputs/`):** Dynamic directory where the system saves generated plots, the consolidated Excel report (`final_report_IPP.xlsx`), and the executive PDF summary (`Resumen_Recomendaciones_IPP.pdf`).

# Setup & Execution Commands
- **Initial Setup:** Run `setup.bat` on Windows to build the Python `venv` and install `requirements.txt`.
- **Launch Application:** Execute `start_app.bat` or `PPI_Launcher.lnk` to launch the Streamlit server.
- **Manual Launch:** `streamlit run app.py`

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

4. **File Path Management:** All file read/write operations must exclusively use the `get_path` helper function defined in `app.py`. This ensures relative file paths resolve correctly regardless of the directory from which the launcher is executed.