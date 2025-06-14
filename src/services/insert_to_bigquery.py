from pydantic import BaseModel, Field
from typing import List, Dict
from google.cloud import bigquery
import datetime


class InsertRecordData(BaseModel):
    user_id: str = Field(..., description="User ID")
    timestamp: str = Field(..., description="Timestamp of the record")
    experiment_type: str = Field(..., description="Type of the experiment")
    rating: float = Field(..., description="Rating given by the user")
    ingredients: List[Dict] = Field(...,
                                    description="List of ingredients used in the experiment")


def insert_record_to_bigquery(insert_record_data: InsertRecordData, dataset_id: str, table_id: str):
    client = bigquery.Client()

    # Define the full table ID
    table_ref = f"{client.project}.{dataset_id}.{table_id}"

    # Insert the sample_record into the BigQuery table
    errors = client.insert_rows_json(
        table_ref,
        [insert_record_data.model_dump()]
    )
    if errors:
        print("Encountered errors while inserting rows: {}".format(errors))
    else:
        print("Insert successful.")


# Delete all records from the BigQuery table before inserting new ones
def delete_all_records_from_bigquery(dataset_id: str, table_id: str):
    client = bigquery.Client()
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    # query = f"DELETE FROM `{table_ref}` WHERE TRUE"
    query = f"TRUNCATE TABLE `{table_ref}`"
    client.query(query).result()
    print("All records deleted from the table.")


def count_record_from_bigquery(dataset_id: str, table_id: str):
    client = bigquery.Client()
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    query = f"SELECT COUNT(*) as cnt FROM `{table_ref}`"
    result = client.query(query).result()
    row_count = next(result).cnt
    print(f"Total records in the table: {row_count}")
    return row_count


if __name__ == "__main__":
    dataset_id = "taste_logs"
    table_id = "coffee_blend_logs"

    import pandas as pd
    from src.config import DATA_DIR

    df = pd.read_csv(DATA_DIR / "input_test.csv")

    row_count = count_record_from_bigquery(dataset_id, table_id)

    if row_count > 0:
        print(
            f"Deleting {row_count} existing records from BigQuery table {dataset_id}.{table_id}.")
        delete_all_records_from_bigquery(dataset_id, table_id)
    else:
        print(
            "No existing records found in BigQuery table. Proceeding to insert new records.")

    for i in range(len(df)):
        insert_record_data = InsertRecordData(
            user_id="hdw",
            timestamp=datetime.datetime.now().isoformat(),
            experiment_type="coffee_001",
            rating=df.iloc[i]["美味しさ"].item(),
            ingredients=[
                {"材料名": col, "分量": df.iloc[i][col].item()} for col in df.columns if col not in ["美味しさ"]
            ]
        )
        insert_record_to_bigquery(insert_record_data, dataset_id, table_id)

    print("All records inserted successfully.")
