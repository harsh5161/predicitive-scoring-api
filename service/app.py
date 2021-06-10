import flask
import pandas as pd
import numpy as np
import json
from flask import request, jsonify
import joblib
from pathlib import Path
from pymongo import MongoClient
from bson.objectid import ObjectId
from pprint import pprint as pp

from engineerings import *
from config import STORAGE_DIR, MONGO_URI


mon = MongoClient(MONGO_URI)
db = mon.get_database()

app = flask.Flask(__name__)
# df = pd.read_csv("datasets/messy9.csv")
# required_frame = pd.DataFrame(
#     df['Latitude-Longitude'])
# required_frame.dropna(axis=0, inplace=True, how='any')
# print(f"length of the dataframe {len(required_frame)}")
if __name__ == '__main__':
    print("running API...")
# data = required_frame.to_json(orient="split")
# dataframe = required_frame.to_dict(orient='records')
# lda_models = joblib.load("model_info")["lda_models"]

def verify(api_key, report_id):
    key = db.apikeys.find_one({ 'value': api_key })
    try:
        return key is not None and key.get('report') == ObjectId(report_id)
    except:
        return False


@app.route('/<report_id>/verify', methods=['POST'])
def verification_route(report_id):
    if not verify(request.headers.get('X-API-Key'), report_id):
        return jsonify({ 'msg': 'Invalid key!' }), 401
    return jsonify({ 'msg': 'OK' }), 200


@app.route('/<report_id>/numeric', methods=['POST'])
def numericengineering(report_id):
    if not verify(request.headers.get('X-API-Key'), report_id):
        return jsonify({ 'msg': 'Invalid key!' }), 401

    res = request.get_json()
    data = json.loads(res['data'])
    X_test = pd.DataFrame.from_dict(data)
    X_test = numeric_engineering(X_test)

    return X_test.to_json(orient="records")


@app.route('/<report_id>/latlong', methods=['GET', 'POST'])
def latlongengineering(report_id):
    if not verify(request.headers.get('X-API-Key'), report_id):
        return jsonify({ msg: 'Invalid key!' }), 401

    res = request.get_json()
    data = json.loads(res['data'])
    lat = json.loads(res['lat'])
    lon = json.loads(res['lon'])
    lat_lon_cols = json.loads(res['lat_lon_cols'])
    X_test = pd.DataFrame.from_dict(data)

    LAT_LONG_DF = latlongEngineering(X_test, lat, lon, lat_lon_cols)
    result = LAT_LONG_DF.to_json(orient="records")

    return result
    # passed_object = {"data": json.dumps(dataframe), "validation": True, "lat": json.dumps(
    #     []), "lon": json.dumps([]), "lat_lon_cols": json.dumps(['Latitude-Longitude'])}
    # return passed_object
    # return json.dumps(dataframe)


@app.route('/<report_id>/url', methods=['POST'])
def urlengineering(report_id):
    if not verify(request.headers.get('X-API-Key'), report_id):
        return jsonify({ msg: 'Invalid key!' }), 401

    res = request.get_json()
    data = json.loads(res['data'])
    X_test = pd.DataFrame.from_dict(data)
    URL_DF = URlEngineering(X_test)

    result = URL_DF.to_json(orient="records")
    return result


@app.route('/<report_id>/date', methods=['POST'])
def dateengineering(report_id):
    if not verify(request.headers.get('X-API-Key'), report_id):
        return jsonify({ msg: 'Invalid key!' }), 401

    res = request.get_json()
    data = json.loads(res['data'])
    possible_datecols = json.loads(res['possible_datecols'])
    possibleDateTimeCols = json.loads(res['possibleDateTimeCols'])
    X_test = pd.DataFrame.from_dict(data)
    DATE_DF = date_engineering(
        X_test, possible_datecols, possibleDateTimeCols, validation=True)

    result = DATE_DF.to_json(orient="records")
    return result


@app.route('/<report_id>/email', methods=['POST'])
def emailengineering(report_id):
    if not verify(request.headers.get('X-API-Key'), report_id):
        return jsonify({ msg: 'Invalid key!' }), 401

    res = request.get_json()
    data = json.loads(res['data'])
    X_test = pd.DataFrame.from_dict(data)
    EMAIL_DF = emailUrlEngineering(
        X_test, email=True, validation=True)

    result = EMAIL_DF.to_json(orient="records")
    return result


@app.route('/<report_id>/sentiment', methods=['POST'])
def sentiment(report_id):
    if not verify(request.headers.get('X-API-Key'), report_id):
        return jsonify({ msg: 'Invalid key!' }), 401

    res = request.get_json()
    data = json.loads(res['data'])
    X_test = pd.DataFrame.from_dict(data)
    sentiment_frame = sentiment_analysis(X_test)

    result = sentiment_frame.to_json(orient="records")
    return result


@app.route('/<report_id>/topic', methods=['POST'])
def topicextraction(report_id):
    if not verify(request.headers.get('X-API-Key'), report_id):
        return jsonify({ msg: 'Invalid key!' }), 401

    rep = db.reports.find_one({ '_id': ObjectId(report_id) })
    dataDir = rep.get('dataDir')
    pp(rep)
    lda_models = joblib.load(Path(STORAGE_DIR) / dataDir / "model_info")["lda_models"]
    res = request.get_json()
    data = json.loads(res['data'])
    index = json.loads(res['index'])
    X_test = pd.DataFrame.from_dict(data)
    topic_frame, _ = topicExtraction(
        X_test, True, lda_models['Model'][index])

    result = topic_frame.to_json(orient="records")
    return result


if __name__ == '__main__':
    app.run(debug=True, port=5001)
