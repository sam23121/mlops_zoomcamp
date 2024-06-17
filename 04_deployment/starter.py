import pickle
import pandas as pd
import numpy as np
import sys




def read_model():
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    return dv, model





def read_data(filename, ls_catgory_cols):
    df = pd.read_parquet(filename)
    
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[ls_catgory_cols] = df[ls_catgory_cols].fillna(-1).astype('int').astype('str')
    
    return df




if __name__ == '__main__':

    year = int(sys.argv[1])
    month = int(sys.argv[2])
    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-0{month}.parquet', categorical)
    dv, model = read_model()

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    print("the mean value for the predictions on april 2024 ", np.mean(y_pred))





    # df["pred"] = y_pred




    # df_result = df[["ride_id", "pred"]]




    # df_result.head()




    # output_file = "predicitions.parquet"
    # df_result.to_parquet(
    #     output_file,
    #     engine='pyarrow',
    #     compression=None,
    #     index=False
    # )






