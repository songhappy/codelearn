import time

import dask
import dask.dataframe as dd
import numpy as np
from dask.distributed import Client, wait
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve, auc, log_loss


class MTE_one_shot:

    def __init__(self, folds, smooth, seed=42):
        self.folds = folds
        self.seed = seed
        self.smooth = smooth

    def fit_transform(self, train, x_col, y_col, y_mean=None, out_col=None, out_dtype=None):

        self.y_col = y_col
        np.random.seed(self.seed)

        if 'fold' not in train.columns:
            fsize = len(train) // self.folds
            train['fold'] = 1
            train['fold'] = train['fold'].cumsum()
            train['fold'] = train['fold'] // fsize
            train['fold'] = train['fold'] % self.folds

        if out_col is None:
            tag = x_col if isinstance(x_col, str) else '_'.join(x_col)
            out_col = f'TE_{tag}_{self.y_col}'

        if y_mean is None:
            y_mean = train[y_col].mean()  # .compute().astype('float32')
        self.mean = y_mean

        cols = ['fold', x_col] if isinstance(x_col, str) else ['fold'] + x_col

        agg_each_fold = train.groupby(cols).agg({y_col: ['count', 'sum']}).reset_index()
        agg_each_fold.columns = cols + ['count_y', 'sum_y']

        agg_all = agg_each_fold.groupby(x_col).agg({'count_y': 'sum', 'sum_y': 'sum'}).reset_index()
        cols = [x_col] if isinstance(x_col, str) else x_col
        agg_all.columns = cols + ['count_y_all', 'sum_y_all']

        agg_each_fold = agg_each_fold.merge(agg_all, on=x_col, how='left')
        agg_each_fold['count_y_all'] = agg_each_fold['count_y_all'] - agg_each_fold['count_y']
        agg_each_fold['sum_y_all'] = agg_each_fold['sum_y_all'] - agg_each_fold['sum_y']
        agg_each_fold[out_col] = (agg_each_fold['sum_y_all'] + self.smooth * self.mean) / (
            agg_each_fold['count_y_all'] + self.smooth)
        agg_each_fold = agg_each_fold.drop(['count_y_all', 'count_y', 'sum_y_all', 'sum_y'], axis=1)

        agg_all[out_col] = (agg_all['sum_y_all'] + self.smooth * self.mean) / (
            agg_all['count_y_all'] + self.smooth)
        agg_all = agg_all.drop(['count_y_all', 'sum_y_all'], axis=1)
        self.agg_all = agg_all

        cols = ['fold', x_col] if isinstance(x_col, str) else ['fold'] + x_col
        train = train.merge(agg_each_fold, on=cols, how='left')
        del agg_each_fold
        # self.agg_each_fold = agg_each_fold
        # train[out_col] = train.map_partitions(lambda cudf_df: cudf_df[out_col].nans_to_nulls())
        train[out_col] = train[out_col].fillna(self.mean)

        if out_dtype is not None:
            train[out_col] = train[out_col].astype(out_dtype)
        return train

    def transform(self, test, x_col, out_col=None, out_dtype=None):
        if out_col is None:
            tag = x_col if isinstance(x_col, str) else '_'.join(x_col)
            out_col = f'TE_{tag}_{self.y_col}'
        test = test.merge(self.agg_all, on=x_col, how='left')
        test[out_col] = test[out_col].fillna(self.mean)
        if out_dtype is not None:
            test[out_col] = test[out_col].astype(out_dtype)
        return test


class FrequencyEncoder:

    def __init__(self, seed=42):
        self.seed = seed

    def fit_transform(self, train, x_col, c_col=None, out_col=None):
        np.random.seed(self.seed)
        if c_col is None or c_col not in train.columns:
            c_col = 'dummy'
            train[c_col] = 1
            drop = True
        else:
            drop = False

        if out_col is None:
            tag = x_col if isinstance(x_col, str) else '_'.join(x_col)
            out_col = f'CE_{tag}_norm'

        cols = [x_col] if isinstance(x_col, str) else x_col
        agg_all = train.groupby(cols).agg({c_col: 'count'}).reset_index()
        if drop:
            train = train.drop(c_col, axis=1)
        agg_all.columns = cols + [out_col]
        agg_all[out_col] = agg_all[out_col].astype('int32')
        agg_all[out_col] = agg_all[out_col] * 1.0 / len(train)
        agg_all[out_col] = agg_all[out_col].astype('float32')

        train = train.merge(agg_all, on=cols, how='left')
        del agg_all
        # print(train.columns)
        # train[out_col] = train.map_partitions(lambda cudf_df: cudf_df[out_col].nans_to_nulls())
        return train

    def transform(self, test, x_col, c_col=None, out_col=None):
        return self.fit_transform(test, x_col, c_col, out_col)


class CountEncoder:

    def __init__(self, seed=42):
        self.seed = seed

    def fit_transform(self, train, test, x_col, out_col=None):
        np.random.seed(self.seed)

        common_cols = [i for i in train.columns if i in test.columns and i != x_col]

        if len(common_cols):
            c_col = common_cols[0]
            drop = False
        else:
            c_col = 'dummy'
            train[c_col] = 1
            test[c_col] = 1
            drop = True

        if out_col is None:
            tag = x_col if isinstance(x_col, str) else '_'.join(x_col)
            out_col = f'CE_{tag}_norm'

        cols = [x_col] if isinstance(x_col, str) else x_col
        agg_all = train.groupby(cols).agg({c_col: 'count'}).reset_index()
        agg_all.columns = cols + [out_col]

        agg_test = test.groupby(cols).agg({c_col: 'count'}).reset_index()
        agg_test.columns = cols + [out_col + '_test']
        agg_all = agg_all.merge(agg_test, on=cols, how='left')
        agg_all[out_col + '_test'] = agg_all[out_col + '_test'].fillna(0)
        agg_all[out_col] = agg_all[out_col] + agg_all[out_col + '_test']
        agg_all = agg_all.drop(out_col + '_test', axis=1)
        del agg_test

        if drop:
            train = train.drop(c_col, axis=1)
            test = test.drop(c_col, axis=1)
        train = train.merge(agg_all, on=cols, how='left')
        test = test.merge(agg_all, on=cols, how='left')
        del agg_all
        return train, test


def diff_encode_cudf_v1(train, col, tar, sft=1):
    train[col + '_sft'] = train[col].shift(sft)
    train[tar + '_sft'] = train[tar].shift(sft)
    out_col = f'DE_{col}_{tar}_{sft}'
    train[out_col] = train[tar] - train[tar + '_sft']
    mask = '__MASK__'
    train[mask] = train[col] == train[col + '_sft']
    train = train.drop([col + '_sft', tar + '_sft'], axis=1)
    train[out_col] = train[out_col] * train[mask]
    train = train.drop(mask, axis=1)
    return train


def diff_language(df, df_lang_count):
    df = df.merge(df_lang_count, how='left', left_on='enaging_user_id',
                  right_on='engaged_with_user_id')
    df['nan_language'] = df['top_language'].isnull()
    df['same_language'] = df['language'] == df['top_language']
    df['diff_language'] = df['language'] != df['top_language']
    df['same_language'] = df['same_language'] * (1 - df['nan_language'])
    df['diff_language'] = df['diff_language'] * (1 - df['nan_language'])
    df = df.drop('top_language', axis=1)
    return df


def compute_prauc(pred, gt):
    prec, recall, thresh = precision_recall_curve(gt, pred)
    prauc = auc(recall, prec)
    return prauc


def compute_AP(pred, gt):
    return average_precision_score(gt, pred)


def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive / float(len(gt))
    return ctr


def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0


# FAST METRIC FROM GIBA
def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)
    strawman_cross_entropy = -(yt * np.log(yt) + (1 - yt) * np.log(1 - yt))
    return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0


def prepare_data(label_names):
    client = Client(n_workers=8, threads_per_worker=10, memory_limit='90GB')

    path = '/Users/guoqiong/intelWork/data/tweet/recsys2021_jennie/train_parquet'
    train = dd.read_parquet(f'{path}/*.parquet')
    print(type(train))

    cols_drop = ['present_links', 'text_tokens', 'tweet']
    train = train.drop(cols_drop, axis=1)

    for col in train.columns:
        if col in label_names:
            train[col] = train[col].astype('float32')
        elif train[col].dtype == 'int64':
            train[col] = train[col].astype('int32')
        elif train[col].dtype == 'int16':
            train[col] = train[col].astype('int8')

    train = train.reset_index(drop=True)

    for col in ['engage_time', 'tweet_timestamp']:
        train[col] = train[col].astype('int64') / 1e9

    train, = dask.persist(train)

    def set_nan(ds):
        mask = ds == 0  # find location where ds is 0?
        ds.loc[mask] = np.nan
        return ds

    train['engage_time'] = train['engage_time'].map_partitions(set_nan)

    train['elapsed_time'] = train['engage_time'] - train['tweet_timestamp']
    train['elapsed_time'] = train.elapsed_time.astype('float64')

    print(train['elapsed_time'].min().compute(), train['elapsed_time'].max().compute())
    print(train['elapsed_time'].mean().compute())

    # # Feature Engineering

    # TRAIN FIRST 5 DAYS. VALIDATE LAST 2 DAYS
    VALID_DOW = [1, 2]  # order is [3, 4, 5, 6, 0, 1, 2]
    valid = train[train['dt_dow'].isin(VALID_DOW)].reset_index(drop=True)
    train = train[~train['dt_dow'].isin(VALID_DOW)].reset_index(drop=True)

    train, valid = dask.persist(train, valid)
    print(type(train), train.shape, valid.shape)

    train = train.set_index('tweet_timestamp')
    valid = valid.set_index('tweet_timestamp')
    train, valid = dask.persist(train, valid)
    print(train.head())

    train = train.reset_index()
    valid = valid.reset_index()
    train, valid = dask.persist(train, valid)
    print(train.head())

    # ### Target Encode

    # TE_media_reply 17.8 seconds<br>
    # TE_tweet_type_reply 27.1 seconds<br>
    # TE_language_reply 52.5 seconds<br>
    # TE_a_user_id_reply 180.0 seconds<br>

    idx = 0;
    cols = []
    start = time.time()
    for t in ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp',
              'like_timestamp']:
        start = time.time()
        for c in ['present_media', 'tweet_type', 'language', 'engaged_with_user_id',
                  'enaging_user_id']:
            out_col = f'TE_{c}_{t}'
            encoder = MTE_one_shot(folds=5, smooth=20)
            train = encoder.fit_transform(train, c, t, out_col=out_col, out_dtype='float32')
            valid = encoder.transform(valid, c, out_col=out_col, out_dtype='float32')
            cols.append(out_col)
            train, valid = dask.persist(train, valid)
            del encoder
            # train.head()
            wait(train)
            wait(valid)
            print(out_col, "%.1f seconds" % (time.time() - start))

    print(train['fold'].value_counts().compute())

    # ### Multiple Column Target Encode

    # cuDF TE ENCODING IS SUPER FAST!!
    idx = 0;
    cols = []
    c = ['present_domains', 'language', 'engagee_follows_engager', 'tweet_type', 'present_media',
         'engaged_with_user_is_verified']
    for t in ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp',
              'like_timestamp']:
        out_col = f'TE_multi_{t}'
        encoder = MTE_one_shot(folds=5, smooth=20)
        train = encoder.fit_transform(train, c, t, out_col=out_col, out_dtype='float32')
        valid = encoder.transform(valid, c, out_col=out_col, out_dtype='float32')
        cols.append(out_col)
        del encoder

    train, valid = dask.persist(train, valid)
    wait(train)
    wait(valid)

    # ### Elapsed Time Target Encode

    # cuDF TE ENCODING IS SUPER FAST!!
    start = time.time()
    idx = 0
    cols = []
    for c in ['present_media', 'tweet_type', 'language']:  # , 'a_user_id', 'b_user_id']:
        for t in ['elapsed_time']:
            out_col = f'TE_{c}_{t}'
            encoder = MTE_one_shot(folds=5, smooth=20)
            train = encoder.fit_transform(train, c, t, out_col=out_col)
            out_dtype = 'float32'  # if 'user_id' in c else None
            valid = encoder.transform(valid, c, out_col=out_col, out_dtype=out_dtype)
            cols.append(out_col)
            print(out_col, "%.1f seconds" % (time.time() - start))
            # del encoder

    train, valid = dask.persist(train, valid)
    wait(train)
    wait(valid)

    # ### Count Encode

    # cuDF CE ENCODING IS SUPER FAST!!
    start = time.time()
    idx = 0;
    cols = []
    for c in ['present_media', 'tweet_type', 'language', 'engaged_with_user_id', 'enaging_user_id']:
        encoder = CountEncoder()
        out_col = f'CE_{c}'
        train, valid = encoder.fit_transform(train, valid, c, out_col=out_col)
        print
        del encoder
        train, valid = dask.persist(train, valid)
        wait(train)
        wait(valid)
        print(out_col, "%.1f seconds" % (time.time() - start))

    # cuDF CE ENCODING IS SUPER FAST!!
    idx = 0;
    cols = []
    start = time.time()
    for c in ['present_media', 'tweet_type', 'language', 'engaged_with_user_id', 'enaging_user_id']:
        encoder = FrequencyEncoder()
        out_col = f'CE_{c}_norm'
        train = encoder.fit_transform(train, c, c_col='tweet_id', out_col=out_col)
        valid = encoder.transform(valid, c, c_col='tweet_id', out_col=out_col)
        cols.append(out_col)
        del encoder
        train, valid = dask.persist(train, valid)
        wait(train)
        wait(valid)
        print(out_col, "%.1f seconds" % (time.time() - start))

    # ### Difference Encode (Lag Features)

    start = time.time()
    # cuDF DE ENCODING IS FAST!!
    idx = 0;
    cols = [];
    sc = 'tweet_timestamp'
    for c in ['enaging_user_id']:
        for t in ['enaging_user_following_count', 'enaging_user_following_count', 'language']:
            for s in [1, -1]:
                start = time.time()
                train = diff_encode_cudf_v1(train, col=c, tar=t, sft=s)
                valid = diff_encode_cudf_v1(valid, col=c, tar=t, sft=s)
                train, valid = dask.persist(train, valid)
                wait(train)
                wait(valid)
                end = time.time();
                idx += 1
                print('DE', c, t, s, '%.1f seconds' % (end - start))

    # ### Diff Language

    train_lang = train[['engaged_with_user_id', 'language', 'tweet_id']].drop_duplicates()
    valid_lang = valid[['engaged_with_user_id', 'language', 'tweet_id']].drop_duplicates()
    train_lang_count = train_lang.groupby(['engaged_with_user_id', 'language']).agg(
        {'tweet_id': 'count'}).reset_index()
    valid_lang_count = valid_lang.groupby(['engaged_with_user_id', 'language']).agg(
        {'tweet_id': 'count'}).reset_index()
    train_lang_count, valid_lang_count = dask.persist(train_lang_count, valid_lang_count)
    train_lang_count.head()
    del train_lang, valid_lang

    train_lang_count = train_lang_count.merge(valid_lang_count,
                                              on=['engaged_with_user_id', 'language'], how='left')
    train_lang_count['tweet_id_y'] = train_lang_count['tweet_id_y'].fillna(0)
    train_lang_count['tweet_id_x'] = train_lang_count['tweet_id_x'] + train_lang_count['tweet_id_y']
    train_lang_count = train_lang_count.drop('tweet_id_y', axis=1)
    train_lang_count.columns = ['engaged_with_user_id', 'top_language', 'language_count']
    train_lang_count, = dask.persist(train_lang_count)
    train_lang_count.head()

    # error in original notebook
    # ??? train_lang_count = train_lang_count.sort_values(['engaged_with_user_id', 'language_count'])
    train_lang_count = train_lang_count.sort_values(['engaged_with_user_id'])
    train_lang_count = train_lang_count.sort_values(['language_count'])

    train_lang_count['engaged_with_user_shifted'] = train_lang_count['engaged_with_user_id'].shift(
        1)
    train_lang_count = train_lang_count[
        train_lang_count['engaged_with_user_id'] != train_lang_count['engaged_with_user_shifted']]
    # ??? train_lang_count = train_lang_count.drop(['a_user_shifted','language_count'],axis=1)
    train_lang_count = train_lang_count.drop(['engaged_with_user_shifted', 'language_count'],
                                             axis=1)

    train_lang_count.columns = ['engaged_with_user_id', 'top_language']
    train_lang_count, = dask.persist(train_lang_count)
    train_lang_count.head()

    # ## Rate feature

    # follow rate feature
    train['a_ff_rate'] = (train['engaged_with_user_following_count'] / train[
        'engaged_with_user_follower_count']).astype('float32')
    train['b_ff_rate'] = (
        train['enaging_user_follower_count'] / train['enaging_user_following_count']).astype(
        'float32')
    valid['a_ff_rate'] = (valid['engaged_with_user_following_count'] / valid[
        'engaged_with_user_follower_count']).astype('float32')
    valid['b_ff_rate'] = (
        valid['enaging_user_follower_count'] / valid['enaging_user_following_count']).astype(
        'float32')

    train, valid = dask.persist(train, valid)
    wait(train)
    wait(valid)

    # # Summarize Features
    label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp',
                   'like_timestamp']
    DONT_USE = ['tweet_id', 'tweet_timestamp', 'engaged_with_user_account_creation',
                'enaging_user_account_creation', 'engage_time',
                'fold', 'enaging_user_id', 'engaged_with_user_id', 'dt_dow',
                'engaged_with_user_account_creation', 'enaging_user_account_creation',
                'elapsed_time',
                'present_links', 'present_domains']
    DONT_USE += label_names
    features = [c for c in train.columns if c not in DONT_USE]

    RMV = [c for c in DONT_USE if c in train.columns and c not in label_names]

    for col in RMV:
        # print(col, col in train.columns)
        if col in train.columns:
            train = train.drop(col, axis=1)
            train, = dask.persist(train)
            train.head()

    for col in RMV:
        # print(col, col in valid.columns)
        if col in valid.columns:
            valid = valid.drop(col, axis=1)
            valid, = dask.persist(valid, )
            valid.head()

    # # Train Model Validate
    # We will train on random 10% of first 5 days and validation on last 2 days

    # In[58]:

    SAMPLE_RATIO = 0.1
    SEED = 1

    if SAMPLE_RATIO < 1.0:
        print(len(train))
        train = train.sample(frac=SAMPLE_RATIO, random_state=42)
        train, = dask.persist(train)
        train.head()
        print(len(train))

    train = train.compute()
    Y_train = train[label_names]
    train = train.drop(label_names, axis=1)

    features = [c for c in train.columns if c not in DONT_USE]
    print('Using %i features:' % (len(features)), train.shape[1])
    np.asarray(features)

    # In[59]:

    SAMPLE_RATIO = 0.35  # VAL SET NOW SIZE OF TEST SET
    SEED = 1
    if SAMPLE_RATIO < 1.0:
        print(len(valid))
        valid = valid.sample(frac=SAMPLE_RATIO, random_state=42)
        valid, = dask.persist(valid)
        valid.head()
        print(len(valid))

    valid = valid.compute()
    Y_valid = valid[label_names]
    valid = valid.drop(label_names, axis=1)

    if train.columns.duplicated().sum() > 0:
        raise Exception(f'duplicated!: {train.columns[train.columns.duplicated()]}')
    print('no dup :) ')

    for col in train.columns:
        if train[col].dtype == 'bool':
            train[col] = train[col].astype('int8')
            valid[col] = valid[col].astype('int8')

    return train, Y_train, valid, Y_valid


if __name__ == "__main__":
    import xgboost as xgb

    print('XGB Version', xgb.__version__)

    start = time.time()
    very_start = time.time()
    label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp',
                   'like_timestamp']

    xgb_parms = {
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.3,
        'eval_metric': 'logloss',
        'objective': 'binary:logistic',
        'nthread': 40,
        'tree_method': 'hist',
    }

    train, Y_train, valid, Y_valid = prepare_data(label_names)
    print(f'X_train.shape {train.shape}')
    print(f'X_valid.shape {valid.shape}')

    # TRAIN AND VALIDATE

    NROUND = 300
    VERBOSE_EVAL = 50
    # ESR = 50

    oof = np.zeros((len(valid), len(label_names)))
    preds = []

    for i in range(4):
        name = label_names[i]
        print('#' * 25)
        print('###', name)
        print('#' * 25)

        start = time.time()
        print('Creating DMatrix...')

        dtrain = xgb.DMatrix(data=train, label=Y_train.iloc[:, i])
        dvalid = xgb.DMatrix(data=valid, label=Y_valid.iloc[:, i])
        print('Took %.1f seconds' % (time.time() - start))

        start = time.time()
        print('Training...')
        model = xgb.train(xgb_parms,
                          dtrain=dtrain,
                          # evals=[(dtrain,'train'),(dvalid,'valid')],
                          num_boost_round=NROUND,
                          # early_stopping_rounds=ESR,
                          verbose_eval=VERBOSE_EVAL)
        print('Took %.1f seconds' % (time.time() - start))

        start = time.time()
        print('Predicting...')
        # Y_valid[f'pred_{name}'] = xgb.dask.predict(client,model,valid)
        oof[:, i] += model.predict(dvalid)
        # preds.append(xgb.dask.predict(client,model,valid))
        print('Took %.1f seconds' % (time.time() - start))

        del model, dtrain, dvalid

    yvalid = Y_valid[label_names].values

    # # Compute Validation Metrics

    txt = ''
    for i in range(4):
        ap = compute_AP(oof[:, i], yvalid[:, i])
        rce = compute_rce_fast(oof[:, i], yvalid[:, i])
        txt_ = f"{label_names[i]:20} AP:{ap:.5f} RCE:{rce:.5f}"
        print(txt_)
        txt += txt_ + '\n'

    print('This notebook took %.1f minutes' % ((time.time() - very_start) / 60.))
