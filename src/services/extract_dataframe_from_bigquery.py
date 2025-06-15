from google.cloud import bigquery
import pandas as pd


def extract_dataframe_from_bigquery(
    dataset_id: str,
    table_id: str,
    user_id: str,
    experiment_type: str
) -> pd.DataFrame:
    """
    Extracts data from BigQuery using the provided SQL query and returns it as a pandas DataFrame.

    Returns:
        pd.DataFrame: The result of the query as a pandas DataFrame.
    """
    client = bigquery.Client()
    query_job = client.query(
        f"""
        SELECT *
        FROM `{client.project}.{dataset_id}.{table_id}`
        WHERE user_id = @user_id
        AND experiment_type = @experiment_type
        """,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                bigquery.ScalarQueryParameter(
                    "experiment_type", "STRING", experiment_type)
            ]
        )
    )
    results = query_job.result()

    # Convert the results to a pandas DataFrame
    df = results.to_dataframe()

    list_expanded_ingredients = df["ingredients"].apply(
        lambda ings: {d["材料名"]: d["分量"] for d in ings}
    ).tolist()
    df_expanded = pd.DataFrame(
        list_expanded_ingredients
    ).fillna(0.0)

    df_expanded["美味しさ"] = df["rating"]

    return df_expanded


if __name__ == "__main__":
    # Example usage
    dataset_id = "taste_logs"
    table_id = "coffee_blend_logs"
    user_id = "hdw"
    experiment_type = "coffee_001"
    df = extract_dataframe_from_bigquery(
        dataset_id=dataset_id,
        table_id=table_id,
        user_id=user_id,
        experiment_type=experiment_type
    )
    print(df.head())
