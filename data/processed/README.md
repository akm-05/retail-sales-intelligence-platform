# data/processed/

The CSVs in this folder are a **small synthetic sample** (400 sales
records, 4 years), included only so `streamlit run app.py` produces a
working demo immediately after cloning, with no Postgres setup and no need
to source the real dataset first.

They are placeholders, not the real Superstore dataset. To generate the
real, full processed tables, place the actual dataset at
`data/raw/superstore.csv` (see `data/raw/README.md`) and run:

```bash
python -m src.run_pipeline
```

This will overwrite every file in this folder with the real pipeline
output (and load the same data into Postgres, if reachable).
