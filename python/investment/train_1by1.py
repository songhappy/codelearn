import datetime
from sklearn.metrics import mean_squared_log_error
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
from codelearn.investment.preprocess_utils  import download_yfinance_data, get_df

# Variables and inputs
prediction_step = 1
TESTDATASIZE = 1
UNROLL_LENGTH = 30
yahoo_etf_input = "/home/arda/intelWork/data/yahoo_stock/yahoo_data.csv"
features_todo= ["COIN","DPST", "^SPX", "IXIC", "^DJI", "^RUT","SPY", "QQQ", "IWM","SOXX"]
features_todo = list(set([feature.replace("^","").lower() for feature in features_todo]))
targets = [feature + "_close" for feature in features_todo]

features_columns1 = [feature + "_close" for feature in features_todo]
features_columns2 = [feature + "_open" for feature in features_todo]
features_columns3 = [feature + "_high" for feature in features_todo]
features_columns4 = [feature + "_low" for feature in features_todo]
features_columns5 = [feature + "_volume" for feature in features_todo]
features_columns6 = [feature + "_close_open" for feature in features_todo]
features_columns7 = [feature + "_high_low" for feature in features_todo]
features_columns = features_columns1 + features_columns2 + features_columns3 + features_columns4 + features_columns5


def get_more_features(df):
    for feature in features_todo:
        try:
            df[f"{feature}_close_open"] = df[f"{feature}_close"] - df[f"{feature}_open"]
            df[f"{feature}_high_low"] = df[f"{feature}_high"] - df[f"{feature}_low"]
        except KeyError as e:
            print(f"Missing columns for {feature}: {e}")
    return df

def scale_1c(df, scaler=preprocessing.MinMaxScaler(), column_name=None):
    if column_name:
        df_scaled = pd.DataFrame(scaler.fit_transform(df[[column_name]]), columns=[column_name])
    else:
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    if type(scaler) == preprocessing.MinMaxScaler:
        print(f"Column {column_name} scaled.", scaler.data_min_[0], scaler.data_max_[0])
    elif type(scaler) == preprocessing.StandardScaler:
        print(f"Column {column_name} scaled.", scaler.mean_[0], scaler.var_[0])
    else:
        print(f"Column {column_name} scaled.")
    return df_scaled, scaler

# Scaling function for X, column by column
def scale_x(df, scaler):
    scalers = {}
    df_scaled = pd.DataFrame()
    for col_name in df.columns:
        scaled_column, scaler = scale_1c(df, scaler=scaler, column_name=col_name)
        df_scaled[col_name] = scaled_column.to_numpy().flatten()
        scalers[col_name] = scaler

    return df_scaled, scalers

# Scaling function for Y
def scale_y(y, scaler):
    y_scaled, scaler = scale_1c(y, scaler=scaler, column_name=None)
    return y_scaled, scaler

# Unrolling function
def unroll(data, unroll_length):
    result = []
    for index in range(len(data) - unroll_length):
        result.append(data[index: index + unroll_length])
    return np.asarray(result)

# Generate train/test sets for X and Y
def generate_train_test(df, target, testdatasize, unroll_length):
    y_train = df[target].to_numpy()[unroll_length:-testdatasize]
    y_test = df[target].to_numpy()[-testdatasize + 1:]
    print("y shape", y_train.shape, y_test.shape)
    x_unrolled = unroll(df.to_numpy(), unroll_length)
    x_train = x_unrolled[:-testdatasize]
    x_test = x_unrolled[-testdatasize:-1]
    print("x shape", x_train.shape, x_test.shape)
    return x_train, y_train, x_test, y_test

# XGBoost Model Function
def xgb_one(x_train, y_train, x_test, y_test, features):
    x_train = x_train.reshape(x_train.shape[0], -1)
    # x_test = x_test.reshape(x_test.shape[0], -1)
    features = features.reshape(1, -1)
    
    model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model_xgb.fit(x_train, y_train)
    # y_hat = model_xgb.predict(x_test)

    # y_test_clipped = np.clip(y_test, a_min=1e-9, a_max=None)
    # y_hat_clipped = np.clip(y_hat, a_min=1e-9, a_max=None)

    # # Calculate RMSLE
    # RMSE = np.sqrt(mean_squared_log_error(y_test_clipped, y_hat_clipped))
    RMSE = None
    predictions = model_xgb.predict(features)

    return predictions, RMSE

# Main script execution
if __name__ == '__main__':
    today = datetime.date.today()
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    tickers = ["COIN","DPST", "^SPX", "^IXIC", "^DJI", "^RUT","SPY", "QQQ", "IWM","SOXX", "SPY", "IVV", "VOO", "QQQ", "IWM", "VTI", "XLF", "XLC", "XLB", "XLE", "XLY", "XLP", "EEM", "EFA", "VEA", "VWO", "AGG", "BND", "LQD", "GLD", "SLV", "DBC", "ARKK", "TAN", "BOTZ"]
    data_file ="/home/arda/intelWork/data/yahoo_stock/yahoo_data.csv"
    data = download_yfinance_data(tickers, start_date="2022-01-01", end_date=tomorrow, data_file=data_file)
    features_columns = features_columns1 + features_columns2 + features_columns3 + features_columns4 + features_columns5
    df_in = get_df(data_file, length=365, features_columns=features_columns)
    df_f = get_more_features(df_in)
    print(df_f.tail(5))
    features_columns = features_columns1 + features_columns2 + features_columns3 + features_columns4 + features_columns5 + features_columns6 + features_columns7
    scaler = preprocessing.MinMaxScaler()
    df_scaled, scalers = scale_x(df_f[features_columns], scaler=scaler)

    report = []
    for target in targets:
        print("--------------------------target", target)
        print(pd.DataFrame(df_f[target]).tail(5))
        y_scaled, y_scaler = scale_y(pd.DataFrame(df_f[target]), scaler=scaler)
        x_train, y_train, x_test, y_test = generate_train_test(df_scaled, target, TESTDATASIZE, UNROLL_LENGTH)

        features = df_scaled[-UNROLL_LENGTH:].to_numpy()
        print(features.shape)

        predictions_raw, RMSLE = xgb_one(x_train, y_train, x_test, y_test, features)
        predictions = y_scaler.inverse_transform(predictions_raw.reshape(-1, 1))
        print(f"Predicted {target} price: {predictions[0, 0]:.2f}")

        today = df_f[target].iloc[-1]
        direction = (predictions[0, 0] - today) / today

        report.append([target, direction, RMSLE])

    for item in report:
        print(f"Target: {item[0]}, Direction: {item[1]:.6f}, RMSLE: {item[2]}")
