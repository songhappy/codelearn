from codelearn.investment.preprocess_utils import get_df
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

if __name__ == "__main__":
    data_file ="/home/arda/intelWork/data/yahoo_stock/yahoo_data.csv"
    features_todo= ["COIN","DPST", "^SPX", "IXIC", "^DJI", "^RUT","SPY", "QQQ", "IWM","SOXX"]
    features_todo = list(set([feature.replace("^","").lower() for feature in features_todo]))
    features_columns1 = [feature + "_close" for feature in features_todo]
    features_columns2 = [feature + "_open" for feature in features_todo]
    features_columns3 = [feature + "_high" for feature in features_todo]
    features_columns4 = [feature + "_low" for feature in features_todo]
    features_columns5 = [feature + "_volume" for feature in features_todo]
    features_columns6 = [feature + "_close_open" for feature in features_todo]
    features_columns7 = [feature + "_high_low" for feature in features_todo]
    features_columns = features_columns1 + features_columns2 + features_columns3 + features_columns4 + features_columns5
    df_in = get_df(data_file, length=365, features_columns=features_columns)
    print(df_in.head(3))
    print(df_in.tail(3))
    
    def get_rsi(df, feature, window=14):
        delta = df[feature + "_close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_metrics(df, features_todo):
        for feature in features_todo:
            df[f"{feature}_close_open"] = df[f"{feature}_close"] - df[f"{feature}_open"]
            df[f"{feature}_high_low"] = df[f"{feature}_high"] - df[f"{feature}_low"]
            delta = df[feature + "_close"].diff()
            df[f"{feature}_delta"] = delta
            df[f"{feature}_rsi_6"] = get_rsi(df, feature, window=6)
            df[f"{feature}_rsi_12"] = get_rsi(df, feature, window=12)
            df[f"{feature}_rsi_24"] = get_rsi(df, feature, window=24)
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[f"{feature}_scaled_volumn"] = scaler.fit_transform(df[[f"{feature}_volume"]])
            df[f"{feature}_momentum"] = df[f"{feature}_delta"] * df[f"{feature}_scaled_volumn"]
            print(df.head(3))
            print(df.tail(3))
            df.drop(columns=[f"{feature}_low",f"{feature}_high", f"{feature}_open"], inplace=True)
            print(df.head(3))
            print(df.tail(3))
        return df


    def get_z_scores(df, features_todo):
        report = {}
        for feature in features_todo:
            feature_close = df[feature + "_close"]
            feature_mean = feature_close.mean()
            feature_std = feature_close.std()
            z_scores = (feature_close - feature_mean) / feature_std
            df[f"{feature}_z_score"] = z_scores
            report[feature] = {"mean": feature_mean, "std": feature_std, "last_z_score": z_scores.iloc[-1]}
            # print(report)
            print(df.head(3))
        return df, report
    
    def get_momentum(df, features_todo):
        for feature in features_todo:
           
            df[f"{feature}_price_change"] = df[f"{feature}_close"] - df[f"{feature}_close"].shift(1)
        return df


    df_with_rsi = get_metrics(df_in, features_todo)

    # df_with_momentum = get_momentum(df_in, features_todo)   

    # df_with_zscore, report = get_z_scores(df_in, features_todo)
    # print(report)
    # print(df_with_zscore.head(3))
    # print(df_with_zscore.tail(3))


# Create DataFrame
df = pd.DataFrame({
    'values': [-10, -20, 0, 20, 30, 40, 50]
})

# Custom scaling to [-1, 1]
df['scaled_-1_1_custom'] = 2 * (df['values'] - df['values'].min()) / (df['values'].max() - df['values'].min()) - 1

# Using MinMaxScaler to scale to [-1, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
df['scaled_-1_1_sklearn'] = scaler.fit_transform(df[['values']])

print(df)
print(df)