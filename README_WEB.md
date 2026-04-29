# Static Website Local Notes

This repository includes a static capstone showcase site in `web/`.

## Run The Website

From the repository root:

```bash
cd web
python -m http.server 8000
```

Then open `http://localhost:8000`.

There is no framework, package install, or build step for the website itself. Use a local web server instead of opening `index.html` directly, because the page loads JSON, components, and model files from `assets/`.

The website runtime does not require the research Python environment. Python is only used here as a simple local file server.

## Do You Need To Run Python First?

No, not for normal viewing. The committed `web/assets/` files are enough for the website to work locally.

The Python scripts are only needed when you want to regenerate website assets from research outputs, for example after rerunning training, evaluation, or model export. They are not part of the website runtime.

## Website Helper Scripts

Website-specific Python helpers live in `web_tools/` so `Scenario_D2D/` and `Scenario_JSAC/` can stay aligned with the main research branch.

Run these from the repository root:

```bash
python web_tools/export_jsac_for_site.py
python web_tools/export_d2d_for_site.py
python web_tools/export_jsac_live_run.py
python web_tools/export_d2d_live_run.py
```

The scripts write into `web/assets/data/`, `web/assets/images/figures/`, and `web/assets/models/`.

The two `export_*_live_run.py` scripts also require the `onnx` Python package because they export and validate browser-readable ONNX models:

```bash
pip install onnx
```

This is only needed when regenerating Live Run model files. It is not needed to view the already-built website.

## Regenerating Assets From Scratch

The website export scripts read generated research artifacts such as `.pkl` and `.pth` files from `Scenario_JSAC/save_main/`, `Scenario_JSAC/save_test/`, `Scenario_D2D/saves/`, and `Scenario_D2D/saves_QoS/`.

Those heavy artifacts are not expected to be available in a fresh clone. If they are missing:

- The JSON export scripts write stub JSON where possible, so the page can still render an empty state.
- The live-run model export scripts require trained checkpoints and will fail until the corresponding research pipeline has produced them.

To regenerate everything from raw research code, activate the Python environment and run the scenario pipelines first:

```bash
source .venv_wpc/bin/activate

cd Scenario_JSAC
python main.py
python test_JSAC.py

cd ../Scenario_D2D
python main.py
python test_QoS.py
```

Then return to the repository root and run the `web_tools/` export scripts.
