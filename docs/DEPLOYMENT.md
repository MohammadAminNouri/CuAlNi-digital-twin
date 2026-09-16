# Run and deploy the dashboard

## Local setup

Use Python 3.11 or 3.12 and run commands from the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-core.txt
python -m streamlit run app.py
```

This installs the dashboard, characterization, core machine learning, and report
exports. To enable CALPHAD, Scheil, CIF/XRD, and optional boosted models, install
`requirements.txt` instead. Scientific calculations still require the cited
databases, structures, measurements, and calibration described in INPUT_SCHEMAS.md.

## Streamlit Community Cloud

1. Select `MohammadAminNouri/CuAlNi-digital-twin` and the branch containing the fixes.
2. Set the main file path to `app.py`.
3. Under advanced settings, select Python **3.12** (3.11 is also tested).
4. Deploy, or reboot an existing app after the repaired files reach its branch.

The root `requirements.txt` installs the full scientific stack. For a lightweight
deployment, replace its contents with `-r requirements-core.txt`; optional panels
will explain which dependency/input is missing. Do not delete the `src/` directory
or upload its contents as loose files. Streamlit configuration belongs at
`.streamlit/config.toml`.

If deployment fails, open the app's build logs first. An import error mentioning
`application` usually means the folder hierarchy or branch is wrong. A dependency
installation failure must be resolved in the selected Python environment before
the app can start.

## GitHub Actions

The workflow lives at `.github/workflows/tests.yml`. Pushes and pull requests run
the core suite on Python 3.11/3.12 and the complete scientific suite on Python
3.12. Core checks include clicking the Streamlit analysis button and checking the
report download controls, plus importing an installed package outside the checkout.

After the workflow reaches the default branch, **Actions → tests → Run workflow**
also runs it manually. Repository Actions must be enabled. A workflow at the root
of the repository is not discovered by GitHub.

Passing Actions verifies software behavior; Streamlit hosting has its own build
and runtime logs. Actions does not deploy a Community Cloud app by itself.

References: [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[GitHub workflows](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflows).
