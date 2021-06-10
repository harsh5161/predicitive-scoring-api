import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import requests
import joblib
import json
import re
import swifter
from sys import argv, stderr, exit
import os

from Binaries import *
from CREDS import API_LINK, API_KEY

# Verify API key
print('Authenticating...')
res = requests.post(f"{API_LINK}verify",
                    headers={ 'X-API-Key': API_KEY })
if res.status_code != 200:
    print('Invalid API key! Authentication failed.', file=stderr)
    exit(1)
print('Successfully authenticated using API Key!')

init_info = joblib.load("model_info")
if len(argv) < 2:
    dfPath = input('File to score: ').strip()
    # dfPath = "titanic.csv"
else:
    dfPath = os.path.abspath(argv[1])

df, _ = importFile(None, None, dfPath)
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
df = duplicateHandler(df)
# If first few rows contains unnecessary info
df, update = dataHandler(df, None)
df = duplicateHandler(df)  # calling again if dataHandler drops columns
print(df.columns)

# Filter DataFrame based on columns
df = getDF(df, init_info)
if not isinstance(df, pd.DataFrame):  # If Columns don't match,
    print('QUITTING!')  # QUIT by printing what columns don't match/are not found

# Numeric Engineering of DATA
print('\n#### Entering Numeric Engineering ####\n')
dataframe = df.to_dict(orient='records')
res = requests.post(f"{API_LINK}numeric",
                    json={"data": json.dumps(dataframe)},
                    headers={ 'X-API-Key': API_KEY })
if res.status_code == 200:
    df = pd.DataFrame(res.json())
else:
    raise Exception(
        f"Numeric Engineering API Call Exited with status: {res.status_code}")
del dataframe
# This is going to be the JSON object with the Training information


X_test = df
print(X_test)
y_test = pd.Series()

if init_info['KEY']:
    # print("Value of Key from Training is: ",init_info['KEY'])
    # print("Total no. of null values present in the Key: ",df[init_info['KEY']].isna().sum())
    df.dropna(axis=0, subset=[init_info['KEY']], inplace=True)
    # print("NUll values after removal are: ",df[init_info['KEY']].isna().sum())
    kkey = df.dtypes[init_info['KEY']]
    try:
        if df[init_info['KEY']].dtype == np.float64:
            df[init_info['KEY']] = df[init_info['KEY']].astype(int)
    except:
        if kkey.any() == np.float64:             # if the key is float convert it to int
            df[init_info['KEY']] = df[init_info['KEY']
                                      ].iloc[:, 0].astype(int)
    if isinstance(df[init_info['KEY']], pd.DataFrame):
        k_test = df[init_info['KEY']].iloc[:, 0]
    else:
        k_test = df[init_info['KEY']]
    k_test.name = 'S.No'
    X_test = X_test.set_index(init_info['KEY'])
    k_test.index = X_test.index
else:
    k_test = X_test.index
    k_test.name = 'S.No'


lat = init_info['lat']
lon = init_info['lon']
dataframe = X_test.to_dict(orient='records')
lat_lon_cols = init_info['lat_lon_cols']
if (lat and lon) or lat_lon_cols:
    # print('Running Lat-Long Engineering on validation dataset')
    # LAT_LONG_DF = latlongEngineering(X_test, lat, lon, lat_lon_cols)
    res = requests.post(f"{API_LINK}latlong", json={"data": json.dumps(
        dataframe), "lat": json.dumps(lat), "lon": json.dumps(lon), "lat_lon_cols": json.dumps(lat_lon_cols)},
        headers={ 'X-API-Key': API_KEY })
    if res.status_code == 200:
        LAT_LONG_DF = pd.DataFrame.from_dict(res.json())
        print(LAT_LONG_DF)
    else:
        raise Exception(
            f"Lat Long API Call Exited with status: {res.status_code}")
    # LAT_LONG_DF.fillna(0.0,inplace=True)
    # print(LAT_LONG_DF)
else:
    LAT_LONG_DF = pd.DataFrame()

date_cols = init_info['DateColumns']
possible_datecols = init_info['PossibleDateColumns']
possibleDateTimeCols = init_info['possibleDateTimeCols']
if date_cols:
    # print('Runnning Date Engineering on validation dataset')
    sliceframe = X_test[date_cols].to_dict(orient='records')
    res = requests.post(f"{API_LINK}date", json={"data": json.dumps(
        sliceframe), "possible_datecols": json.dumps(possible_datecols), "possibleDateTimeCols": json.dumps(possibleDateTimeCols)},
        headers={ 'X-API-Key': API_KEY })
    if res.status_code == 200:
        DATE_DF = pd.DataFrame.from_dict(res.json())
        print(DATE_DF)
    else:
        raise Exception(
            f"Date API Call Exited with status: {res.status_code}")
    DATE_DF = DATE_DF[init_info['DateFinalColumns']]
    DATE_DF.fillna(init_info['DateMean'], inplace=True)
else:
    DATE_DF = pd.DataFrame()

if init_info['EMAIL_STATUS'] is False:
    email_cols = init_info['email_cols']

    if len(email_cols) > 0:
        # print('Runnning Email Engineering on validation dataset')
        sliceframe = X_test[email_cols].to_dict(orient='records')
        res = requests.post(f"{API_LINK}email", json={
                            "data": json.dumps(sliceframe)},
                            headers={ 'X-API-Key': API_KEY })
        if res.status_code == 200:
            EMAIL_DF = pd.DataFrame.from_dict(res.json())
            print(EMAIL_DF)
        else:
            raise Exception(
                f"Email API Call Exited with status: {res.status_code}")
        EMAIL_DF.reset_index(drop=True)
        #EMAIL_DF.fillna('missing', inplace=True)
        # print(EMAIL_DF)
    else:
        EMAIL_DF = pd.DataFrame()
else:
    EMAIL_DF = pd.DataFrame()

url_cols = init_info['url_cols']
if len(url_cols) > 0:
    # print('Running URL Egnineering on validation dataset')
    sliceframe = X_test[url_cols].to_dict(orient='records')
    res = requests.post(f"{API_LINK}url", json={
                        "data": json.dumps(sliceframe)},
                        headers={ 'X-API-Key': API_KEY })
    if res.status_code == 200:
        URL_DF = pd.DataFrame.from_dict(res.json())
        print(URL_DF)
    else:
        raise Exception(
            f"URL API Call Exited with status: {res.status_code}")
    URL_DF.reset_index(drop=True)
    # URL_DF.fillna('missing',inplace=True)
    # print(URL_DF)
else:
    URL_DF = pd.DataFrame()

X_test.reset_index(drop=True, inplace=True)
DATE_DF.reset_index(drop=True, inplace=True)
LAT_LONG_DF.reset_index(drop=True, inplace=True)
EMAIL_DF.reset_index(drop=True, inplace=True)
URL_DF.reset_index(drop=True, inplace=True)
concat_list = [X_test, DATE_DF, LAT_LONG_DF, EMAIL_DF, URL_DF]
X_test = pd.concat(concat_list, axis=1)

if len(init_info['NumericColumns']) != 0:
    num_df = X_test[init_info['NumericColumns']]
    num_df = num_df.swifter.apply(
        lambda x: pd.to_numeric(x, errors='coerce'))
    num_df.fillna(init_info['NumericMean'], inplace=True)
else:
    num_df = pd.DataFrame()

if len(init_info['DiscreteColumns']) != 0:
    disc_df = X_test[init_info['DiscreteColumns']]
    disc_cat = init_info['disc_cat']
    for col in disc_df.columns:
        disc_df[col] = disc_df[col].apply(
            lambda x: x if x in disc_cat[col] else 'others')
    disc_df.fillna('missing', inplace=True)
else:
    disc_df = pd.DataFrame()

if init_info['remove_list'] is not None:
    X_test.drop(columns=init_info['remove_list'], axis=1, inplace=True)

some_list = init_info['some_list']
lda_models = init_info['lda_models']

if some_list:
    # print("The review/comment columns found are", some_list)
    start = time.time()
    sliceframe = X_test[some_list].to_dict(orient='records')
    res = requests.post(f"{API_LINK}sentiment", json={
                        "data": json.dumps(sliceframe)},
                        headers={ 'X-API-Key': API_KEY })
    if res.status_code == 200:
        sentiment_frame = pd.DataFrame.from_dict(res.json())
    else:
        raise Exception(
            f"Sentiment API Call Exited with status: {res.status_code}")
    sentiment_frame.fillna(value=0.0, inplace=True)
    # print(sentiment_frame)
    TEXT_DF = sentiment_frame.copy()
    TEXT_DF.reset_index(drop=True, inplace=True)
    end = time.time()
    # print("Sentiment time",end-start)
    start = time.time()
    new_frame = pd.DataFrame(X_test[some_list].copy())
    new_frame.fillna(value="None", inplace=True)
    ind = 0
    for col in new_frame.columns:
        newslice = pd.DataFrame(new_frame[[col]])
        newslice.columns = [str(col)]
        sliceframe = newslice.to_dict(orient='records')
        res = requests.post(f"{API_LINK}topic", json={"data": json.dumps(
            sliceframe), "index": json.dumps(ind)},
            headers={ 'X-API-Key': API_KEY })
        if res.status_code == 200:
            topic_frame = pd.DataFrame.from_dict(res.json())
            topic_frame.columns = [str(col)+"_Topic"]
            topic_frame.rename(columns={0: str(col)+"_Topic"}, inplace=True)
            topic_frame.reset_index(drop=True, inplace=True)
        else:
            raise Exception(
                f"Topic Extraction API Call Exited with status: {res.status_code}")
        # print(topic_frame)
        TEXT_DF = pd.concat([TEXT_DF, topic_frame], axis=1, sort=False)
        ind = ind+1
    X_test.drop(some_list, axis=1, inplace=True)
    print(TEXT_DF)
else:
    TEXT_DF = pd.DataFrame()
disc_df.reset_index(drop=True, inplace=True)
num_df.reset_index(drop=True, inplace=True)
TEXT_DF.reset_index(drop=True, inplace=True)
if not TEXT_DF.empty:
    for col in TEXT_DF.columns:
        if col.find("_Topic") != -1:
            disc_df = pd.concat(
                [disc_df, pd.DataFrame(TEXT_DF[col])], axis=1)
        else:
            num_df = pd.concat(
                [num_df, pd.DataFrame(TEXT_DF[col])], axis=1)
num_df = num_df[init_info['PearsonsColumns']]
num_df.reset_index(drop=True, inplace=True)
disc_df.reset_index(drop=True, inplace=True)
if num_df.shape[1] != 0:  # Some datasets may contain only categorical data
    X_test = pd.concat([num_df, disc_df], axis=1)
else:
    X_test = disc_df
X_test = init_info['TargetEncoder'].transform(X_test)
X_test = X_test[init_info['TrainingColumns']]
X_test = X_test.fillna(X_test.mode())
mm = init_info['MinMaxScaler']
# Clip the data with training min and max, important
X_test.clip(mm.data_min_, mm.data_max_, inplace=True, axis=1)
X_test = mm.transform(X_test)
X_test = pd.DataFrame(init_info['PowerTransformer'].transform(
    X_test), columns=init_info['TrainingColumns'])
new_mm = MinMaxScaler()
X_test = pd.DataFrame(new_mm.fit_transform(
    X_test), columns=init_info['TrainingColumns'])
mod = init_info['model']
y_pred = mod.predict(X_test)
if init_info['ML'] == 'Classification':
    y_probas = mod.predict_proba(X_test)
    y_pred = pd.Series(
        init_info['TargetLabelEncoder'].inverse_transform(y_pred))
    y_probs_cols = init_info['y_probs_cols']
    y_probas = pd.DataFrame(y_probas, columns=y_probs_cols)


preview_length = len(X_test)

preview = pd.DataFrame({k_test.name: k_test.values.tolist(),
                        'Predicted Values': y_pred.tolist()})

if init_info['ML'] == 'Classification':
    preview = pd.concat([preview, y_probas], axis=1)

    for col in ['Predicted Values']:
        if preview[col].dtype == np.float64:
            preview[col] = preview[col].astype(int)

    yp = {}
    for i in y_probas.columns:
        yp[i] = str(i).replace("Probabilities", "Probability")
        yp[i] = str(i).replace("0.0", "0")
        yp[i] = str(i).replace("1.0", "1")
    preview.rename(columns=yp, inplace=True)       # to rename columns

for col in preview.columns:       # to round off decimal places of large float entries in preview
    if preview[col].dtype == np.float64:
        preview[col] = pd.Series(preview[col]).round(decimals=3)


preview_vals = preview['Predicted Values'].value_counts()
preview.to_csv('score.csv', sep=',', index=False)
print(">>>>>>[[Scoring File Saved]]>>>>>")
print('Output saved to score.csv.')
print('\nCode executed Successfully')
print(">>>>>>[[Scoring Completed]]>>>>>")
