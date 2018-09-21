# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 22:38:15 2018

@author: salman
"""

# scale_pos_weight, predict_proba
import warnings
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from gensim import matutils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Function to get the S&P data
def get_snp_df(csvfile, threshold=20):
    snp_df = pd.read_csv(csvfile)

    # Basic DF manipulation
    snp_df.dropna(subset=["PX_LAST"], inplace=True)
    snp_df = snp_df[["Dates", "PX_LAST"]]
    snp_df["Dates"] = pd.to_datetime(snp_df["Dates"])
    snp_df = snp_df.sort_values("Dates", ascending=1)
    # snp_df['weekday'] = snp_df['Dates'].dt.dayofweek
    # snp_df = snp_df.set_index("Dates")
    # print(snp_df["PX_LAST"].rolling(5).std().head())
    snp_df["Vol"] = snp_df["PX_LAST"].rolling(5).std()
    snp_df["Label"] = 0
    snp_df.loc[snp_df["Vol"] >= threshold, "Label"] = 1

    return snp_df


# Function to get the News data
def get_news_df(news_filename):
    news_df = pd.read_csv(news_filename, sep="\#\%\$\%\$\%\#", engine="python")
    '''with pd.option_context('display.max_rows', None,
                           'display.max_columns', None):
        print(news_df.head())'''

    return news_df


# Function to get the word weights
def get_weight_df(news_df,
                  word2vec_file="GoogleNews-vectors-negative300-SLIM.bin",
                  is_bin=True,
                  num_important=1):
    stop_words = set(stopwords.words('english'))
    matrix = []
    filtered = []

    # Word2vec slim
    model = KeyedVectors.load_word2vec_format(
                'GoogleNews-vectors-negative300-SLIM.bin', binary=True)

    for i, row in news_df.iterrows():
        try:
            filtered_text = [model[w] for w in row['Snippet'].split()
                             if w in model
                             and w not in stop_words]
            filtered.append([w for w in row['Snippet'].split()
                             if w in model
                             and w not in stop_words])

            if len(filtered_text):
                curr_arr = matutils.unitvec(np.array(filtered_text)
                                            .mean(axis=0))
                curr_arr = np.append(curr_arr, row["Date"])
                curr_arr = np.append(curr_arr, row["Section"])
                curr_arr = np.append(curr_arr, row["Category"])
                matrix.append(curr_arr)
        except AttributeError:  # Some error saying row['Snippet'] is int
            continue

    i = 1
    base_str = "feature_"
    col_list = []

    while i <= 300:
        col_list.append(base_str + str(i))
        i += 1

    col_list.append("Date")
    col_list.append("Section")
    col_list.append("Category")
    mat_df = pd.DataFrame(data=matrix, columns=col_list)
    sum_col_list = col_list[:-3]
    # Most important articles by total weight
    mat_df["tot_wt"] = mat_df[sum_col_list].sum(axis=1)
    # Getting the num_important most important articles
    mat_df = mat_df.\
        sort_values(['Date', 'tot_wt'], ascending=[1, 0]).\
        groupby('Date').head(num_important)

    return mat_df


if __name__ == "__main__":
    csvfile = "bbg_data.csv"
    snp_df = get_snp_df(csvfile, threshold=20)

    '''"2008.txt", "2009.txt", "2010.txt",
                          "2011.txt", "2012.txt", "2013.txt",
    "2014.txt", "2015.txt", "2016.txt",'''
    news_filename_list = ["2016.txt", "2017.txt"]
    news_df = pd.DataFrame()

    for news_filename in news_filename_list:
        # print(news_filename)
        curr_df = get_news_df(news_filename)
        news_df = news_df.append(curr_df)

    mat_df = get_weight_df(news_df, num_important=5)
    for index, row in mat_df.iterrows():
        try:
            mat_df["Date"] = pd.to_datetime(mat_df["Date"], format="%Y%m%d")
        except ValueError:  # TODO: Handle this error properly
            # mat_df = mat_df.drop(index)
            continue

    # Merging the news and S&P data
    print("Data cleaning done. Starting process.")
    mat_df = mat_df.merge(snp_df[["Dates", "Label"]], how="left",
                          left_on=["Date"],
                          right_on=["Dates"])
    scale_pos_weight = len(mat_df[mat_df["Label"] == 0]) /\
        len(mat_df[mat_df["Label"] == 1])

    train_set, test_set = train_test_split(
            mat_df, test_size=0.3, random_state=40)
    '''train_set, cv_set = train_test_split(
            mat_df, test_size=0.2, random_state=40)'''
    X_train = train_set.drop(["Label", "Date", "Section", "Dates",
                              "Category", 'tot_wt'],
                             axis=1)

    # Handling strange can't convert errors
    for col in list(X_train):
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')

    y_train = train_set["Label"]
    y_train = pd.to_numeric(y_train, errors='coerce')

    '''X_cv = cv_set.drop(["Label", "Date", "Section", "Dates",
                        "Category", 'tot_wt'],
                       axis=1)

    # Handling strange can't convert errors
    for col in list(X_cv):
        X_cv[col] = pd.to_numeric(X_cv[col], errors='coerce')

    y_cv = cv_set["Label"]
    y_cv = pd.to_numeric(y_cv, errors='coerce')'''

    X_test = test_set.drop(["Label", "Date", "Section", "Dates",
                            "Category", 'tot_wt'],
                           axis=1)

    # Handling strange can't convert errors
    for col in list(X_test):
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    y_test = test_set["Label"]
    y_test = pd.to_numeric(y_test, errors='coerce')
    y_train_new = y_train.replace(np.nan, 0)
    # y_cv_new = y_cv.replace(np.nan, 0)
    y_test_new = y_test.replace(np.nan, 0)

    dtrain = xgb.DMatrix(X_train, label=y_train_new)
    # dcv = xgb.DMatrix(X_cv, label=y_cv_new)
    param = {'max_depth': 6, 'eta': 0.3, 'silent': 1,
             'objective': 'binary:logitraw', 'eval_metric': 'auc',
             'scale_pos_weight': scale_pos_weight}
    # watchlist = [(dcv, 'eval'), (dtrain, 'train')]
    # num_round = 4
    n_estimators = 3000
    # bst = xgb.train(param, dtrain, num_round, watchlist)
    # bst = xgb.XGBClassifier(max_depth=6, n_estimators=n_estimators,
    #                        learning_rate=0.3,
    #                        scale_pos_weight=scale_pos_weight)
    # bst.cv(X_train, y_train_new)
    bst = xgb.XGBClassifier()
    parameters = {'max_depth': [4, 6, 8], 'learning_rate': [0.2, 0.3, 0.4],
                  'scale_pos_weight': [scale_pos_weight]}
    cv = TimeSeriesSplit(n_splits=3)
    clf = GridSearchCV(bst, parameters, cv=cv)
    clf.fit(X_train, y_train_new)
    # print(sorted(clf.cv_results_.keys()))
    y_test_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test_new, y_test_pred)
    roc_auc = roc_auc_score(y_test_new, y_test_pred)
    print(accuracy, roc_auc)

    '''X_pred = mat_df.drop(["Label", "Date", "Section", "Dates",
                          "Category", 'tot_wt'],
                         axis=1)

    for col in list(X_pred):
        X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce')'''

    '''dpred = xgb.DMatrix(X_pred)
    # ypred = bst.predict(dpred, ntree_limit=bst.best_ntree_limit)
    pred_proba = bst.predict_proba(X_test)
    print(pred_proba.shape)'''
    '''out_df = pd.DataFrame()
    out_df["Label"] = ypred
    out_df["Date"] = mat_df["Date"]
    out_df = out_df.groupby("Date").mean().reset_index()'''
    # print(out_df.head(20))
