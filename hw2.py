import mlflow
from mlflow.models import infer_signature
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# по заданию имя эксперимента и родительского рана должны быть заданы как фамилия и имя и как телеграм никнейм
exp_name = "alina_lukmanova"
parent_run_name = "AlinaLukmanova"

# Создадим эксперимент, если его нет, или возьмем его id, если он уже существует.
results = mlflow.search_experiments(filter_string=f"name = '{exp_name}'")
if len(results) > 0:
    exp_id = results[0].experiment_id
else:
    exp_id = mlflow.create_experiment(exp_name)


FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", ]
TARGET = "MedHouseVal"
models = dict(
    zip(["random_forest", "linear_regression", "decision_tree"],
        [RandomForestRegressor, LinearRegression, DecisionTreeRegressor, ]))


# Загрузим датасет и разделим его на тренировочную, валидационную и тестовую выборки.
housing_dataset = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(housing_dataset['data'], housing_dataset['target'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)


# Создадим необходимые запуски
with mlflow.start_run(run_name=parent_run_name, experiment_id=exp_id, description="parent") as parent_run:
    for model_name in models.keys():
        # Запустим child run на каждую модель.
        with mlflow.start_run(run_name=model_name, experiment_id=exp_id, nested=True) as child_run:

            model = models[model_name]()
            model.fit(X_train, y_train)
            prediction = model.predict(X_val)

            # Создадим валидационный датасет.
            eval_df = X_val.copy()
            eval_df["target"] = y_val

            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, "linreg", signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                model_type="regressor",
                evaluators=["default"],
            )