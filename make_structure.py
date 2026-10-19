from pathlib import Path

folders = ["data/raw", "data/processed", "notebooks", "src/data", "src/causal",
           "src/optimization", "src/monitoring", "src/api", "models", "tests", "docs"]

files = ["README.md", ".gitignore", ".env.example", "data/README.md",
         "src/__init__.py", "src/data/__init__.py", "src/causal/__init__.py",
         "src/optimization/__init__.py", "src/monitoring/__init__.py", "src/api/__init__.py",
         "src/data/generate_synthetic.py", "src/data/preprocess.py", "src/data/propensity.py",
         "src/causal/dag.py", "src/causal/dml_model.py", "src/causal/evaluate_uplift.py", "src/causal/refute.py",
         "src/optimization/allocator.py", "src/monitoring/drift.py",
         "src/api/main.py", "src/api/schemas.py", "src/api/services.py",
         "tests/test_data.py", "tests/test_dml.py", "tests/test_allocator.py", "tests/test_api.py",
         "docs/results.md",
         "notebooks/01_eda.ipynb"]

for f in folders:
    Path(f).mkdir(parents=True, exist_ok=True)
for f in files:
    Path(f).touch()

print("Done. Folders and files created.")