import mlflow
import pickle


mlflow.set_tracking_uri("http://mlflow:5000")

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(dv, lr, *args, **kwargs):

    with open('models/lin_reg.bin', 'wb') as f_out:
        pickle.dump((dv, lr), f_out)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")
    # Specify your data exporting logic here



    with mlflow.start_run():

        alpha = 0.1
        mlflow.log_param("alpha", alpha)
        lr = Lasso(alpha)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        mlflow.log_artifact(local_path="models/lin_reg.bin", artifact_path="models_pickle")
    # Specify your data exporting logic here


