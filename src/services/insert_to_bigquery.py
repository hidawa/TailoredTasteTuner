from google.cloud import bigquery
import datetime

client = bigquery.Client()


if __name__ == "__main__":
    dataset_id = "taste_logs"
    table_id = "coffee_blend_logs"

    # Define the full table ID
    table_ref = f"{client.project}.{dataset_id}.{table_id}"

    sample_record = {
        "user_id": "hdw",
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_type": "coffee_001",
        "rating": 6,
        "ingredients": [
            {"材料名": "グアテマラ", "分量": 1},
        ]
    }

    now_str = datetime.datetime.now()
    print(now_str)

    # Insert the sample_record into the BigQuery table
    errors = client.insert_rows_json(table_ref, [sample_record])
    if errors:
        print("Encountered errors while inserting rows: {}".format(errors))
    else:
        print("Insert successful.")
