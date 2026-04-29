# Website Tools

These scripts regenerate the static website assets in `../web/assets/` from research outputs stored under `../Scenario_D2D/` and `../Scenario_JSAC/`.

Run them from the repository root:

```bash
python web_tools/export_jsac_for_site.py
python web_tools/export_d2d_for_site.py
python web_tools/export_jsac_live_run.py
python web_tools/export_d2d_live_run.py
```

The website does not run these scripts at runtime. See `../WEB_README.md` for the local website quick start and the full asset-regeneration note.
