import os

# Root of the data layout described in the README (benchmarks/, raw/, qa_models/).
# Defaults to the copy inside the repository; override with the SABET_DATA_DIR environment variable.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("SABET_DATA_DIR", os.path.join(REPO_ROOT, "Data"))
