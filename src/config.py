from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # /app
DATA_DIR = BASE_DIR / "data"


BQ_DATASET_ID = "taste_logs"
BQ_TABLE_ID = "coffee_blend_logs"
SAMPLE_USER_ID = "hdw"
SAMPLE_EXPERIMENT_TYPE = "coffee_001"
