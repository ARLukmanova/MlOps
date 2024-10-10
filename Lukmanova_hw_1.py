import io
import json
import pickle
from datetime import timedelta, datetime
from typing import Any, Dict, Literal

import pandas as pd
from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Constants
BUCKET = Variable.get("S3_BUCKET")
DEFAULT_ARGS = {
    'owner': 'Lukmanova_Alina',
    'email': 'sun-beam@list.ru',
    'email_on_failure': True,
    'email_on_retry': False,
    'retry': 3,
    'retry_delay': timedelta(minutes=1),
}

FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude",
]
TARGET = "MedHouseVal"

# Models definition
MODEL_CLASSES = {
    "random_forest": RandomForestRegressor,
    "linear_regression": LinearRegression,
    "decision_tree": DecisionTreeRegressor
}

def upload_to_s3(s3_hook, buffer, key: str):
    """Helper function to upload buffer to S3."""
    buffer.seek(0)
    s3_hook.load_file_obj(buffer, key=key, bucket_name=BUCKET, replace=True)

def fetch_dataset() -> pd.DataFrame:
    """Fetch California housing dataset and return as DataFrame."""
    housing = fetch_california_housing(as_frame=True)
    return pd.concat([housing["data"], pd.DataFrame(housing["target"], columns=[TARGET])], axis=1)

def split_and_scale_data(data: pd.DataFrame):
    """Split data into train and test sets, and apply StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(data[FEATURES], data[TARGET], test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

def create_dag(dag_id: str, model_name: str):

    def init() -> Dict[str, Any]:
        return {
            "model_name": model_name,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_data(**kwargs) -> Dict[str, Any]:
        metrics = kwargs['ti'].xcom_pull(task_ids='init')
        data = fetch_dataset()

        # Upload dataset to S3
        s3_hook = S3Hook("s3_connection")
        buffer = io.BytesIO()
        data.to_pickle(buffer)
        upload_to_s3(s3_hook, buffer, f"{DEFAULT_ARGS['owner']}/{model_name}/datasets/Dataset_California.pkl")

        metrics.update({
            "get_dataset_end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_size": {"n_rows": data.shape[0], "n_columns": data.shape[1]}
        })
        return metrics

    def prepare_data(**kwargs) -> Dict[str, Any]:
        metrics = kwargs['ti'].xcom_pull(task_ids='get_data')

        # Download dataset from S3
        s3_hook = S3Hook("s3_connection")
        file = s3_hook.download_file(
            key=f"{DEFAULT_ARGS['owner']}/{model_name}/datasets/Dataset_California.pkl",
            bucket_name=BUCKET
        )
        data = pd.read_pickle(file)

        # Split and scale data
        X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(data)

        # Upload processed data to S3
        buffer = io.BytesIO()
        pickle.dump((X_train_scaled, X_test_scaled, y_train, y_test), buffer)
        upload_to_s3(s3_hook, buffer, f"{DEFAULT_ARGS['owner']}/{model_name}/datasets/Dataset_splited_and_scaled.pkl")

        metrics.update({
            "prepare_data_end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feature_names": FEATURES
        })
        return metrics

    def train_model(**kwargs) -> Dict[str, Any]:
        metrics = kwargs['ti'].xcom_pull(task_ids='prepare_data')

        # Download scaled data from S3
        s3_hook = S3Hook("s3_connection")
        file_path = s3_hook.download_file(
            key=f"{DEFAULT_ARGS['owner']}/{model_name}/datasets/Dataset_splited_and_scaled.pkl",
            bucket_name=BUCKET
        )
        with open(file_path, 'rb') as file:
            X_train, X_test, y_train, y_test = pickle.load(file)

        # Train the model and calculate metrics
        model = MODEL_CLASSES[model_name]()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions, squared=False)

        metrics.update({
            "train_model_end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_metrics": {"r2": r2, "rmse": rmse}
        })
        return metrics

    def save_results(**kwargs) -> None:
        metrics = kwargs['ti'].xcom_pull(task_ids='train_model')

        buffer = io.BytesIO()
        buffer.write(json.dumps(metrics).encode())
        upload_to_s3(S3Hook("s3_connection"), buffer, f"{DEFAULT_ARGS['owner']}/{model_name}/results/metrics.json")

    # Initialize DAG
    dag = DAG(
        dag_id=dag_id,
        schedule_interval='0 1 * * *',
        start_date=days_ago(2),
        catchup=False,
        tags=['mlops'],
        default_args=DEFAULT_ARGS
    )

    with dag:
        task_init = PythonOperator(task_id='init', python_callable=init, dag=dag)
        task_get_data = PythonOperator(task_id='get_data', python_callable=get_data, dag=dag, provide_context=True)
        task_prepare_data = PythonOperator(task_id='prepare_data', python_callable=prepare_data, dag=dag, provide_context=True)
        task_train_model = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag, provide_context=True)
        task_save_results = PythonOperator(task_id='save_results', python_callable=save_results, dag=dag, provide_context=True)

        task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results

    return dag

# Create DAGs for all models
for model_name in MODEL_CLASSES.keys():
    globals()[f"dag_{model_name}"] = create_dag(f"Lukmanova_Alina_{model_name}", model_name)
