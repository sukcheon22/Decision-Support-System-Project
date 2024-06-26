import flask
import joblib
import pandas as pd
from flask import request, jsonify
import numpy as np
from datetime import datetime
import tensorflow as tf

from transformer_model import TransformerModel, predict_stock_price
import torch

date_start_input1 = '1/11/2024'
date_start_input2 = '1/16/2024'
date_start_input3 = '1/11/2024'
date_start_var = '4/30/2024'
data_1 = pd.read_csv('Data_input_1.csv')
data_2 = pd.read_csv('Data_input_2.csv')
data_3 = pd.read_csv('Data_input_3.csv')
# Load pre-trained models
# model = tf.keras.models.load_model('LSTM.h5')
models = {
    'lstm_1layer_econ': [tf.keras.models.load_model('1_layerLSTMEconomic.h5'), joblib.load('scaler_price_economic1lstm.pkl'), np.load('X_test_economic.npy')],
    'lstm_1layer_main': [tf.keras.models.load_model('1_layerLSTMMain.h5'), joblib.load('scaler_price_main1lstm.pkl'), np.load('X_test_main.npy')],
    'lstm_1layer_tech': [tf.keras.models.load_model('1_layerLSTMTechnical.h5'), joblib.load('scaler_price_tech1lstm.pkl'), np.load('X_test_technical.npy')],
    'lstm_2layer_econ': [tf.keras.models.load_model('2_layerLSTMEconomic.h5'), joblib.load('scaler_price_economic2lstm.pkl'), np.load('X_test_economic.npy')],
    'lstm_2layer_main': [tf.keras.models.load_model('2_layerLSTMMain.h5'), joblib.load('scaler_price_main2lstm.pkl'), np.load('X_test_main.npy')],
    'lstm_2layer_tech': [tf.keras.models.load_model('2_layerLSTMTechnical.h5'), joblib.load('scaler_price_technical2lstm.pkl'), np.load('X_test_technical.npy')],
    'lstm_bi_econ': [tf.keras.models.load_model('BiLSTMEconomic.h5'), joblib.load('scaler_price_economicbilstm.pkl'), np.load('X_test_economic.npy')],
    'lstm_bi_main': [tf.keras.models.load_model('BiLSTMMain.h5'), joblib.load('scaler_price_mainbilstm.pkl'), np.load('X_test_main.npy')],
    'lstm_bi_tech': [tf.keras.models.load_model('BiLSTMTechnical.h5'), joblib.load('scaler_price_techbilstm.pkl'), np.load('X_test_technical.npy')]
    
    # 'polynomial_regression': joblib.load('model_poly.sav'),
    # 'decision_tree': joblib.load('model_tree.sav')
}
models_stats = {
    'var_main': [joblib.load('VARMain.pkl'), np.load('forecast_main.npy')],
    'var_tech': [joblib.load('VARTechnical.pkl'), np.load('forecast_tech.npy')],
    'var_econ': [joblib.load('VAREconomic.pkl'), np.load('forecast_economic.npy')]
}
models_trans = {
    'trans_main': [torch.load('TransformerMain.pth'), joblib.load('scaler_price_maintransformer.pkl'), np.load('X_test_maintransformer.npy')],
    'trans_tech': [torch.load('TransformerTech.pth'), joblib.load('scaler_price_techtransformer.pkl'), np.load('X_test_techtransformer.npy')],
    'trans_econ': [torch.load('TransformerEcon.pth'), joblib.load('scaler_price_econtransformer.pkl'), np.load('X_test_econtransformer.npy')]
}
app = flask.Flask(__name__, template_folder='templates')
# app._static_folder = 'static'
@app.route('/')
def main():
    return flask.render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    date = request.json.get('date')
    date = datetime.strptime(date, '%Y-%m-%d')
    month = int(date.strftime('%m'))
    day = int(date.strftime('%d'))
    date = '{}/{}/{}'.format(month, day, date.strftime('%Y'))
    model_name = request.json.get('model')
    
    # date_datetime = datetime.strptime(date, '%Y-%m-%d')
    # date_start_datetime = datetime.strptime(date_start, '%Y-%m-%d')
    # num_date = abs((date_datetime - date_start_datetime).days)

    # if model_name not in models:
    #     return jsonify({'error': 'Invalid model selected'}), 400
    if model_name in models:
        model, scaler_price, X_test = models[model_name]
        
        # Predict using the loaded model
        if np.array_equal(X_test, np.load('X_test_economic.npy')):  
            index_start = data_3['Date'].index[data_3['Date'] == date_start_input3][0]
            index_find = data_3['Date'].index[data_3['Date'] == date][0]
            num_date = index_start - index_find
            prediction = model.predict(X_test[num_date][np.newaxis, :, :])
            prediction = scaler_price.inverse_transform(np.concatenate((prediction, np.zeros((len(prediction), 3))), axis=1))[:,0]
        elif np.array_equal(X_test, np.load('X_test_main.npy')):       
            index_start = data_1['Date'].index[data_1['Date'] == date_start_input1][0]
            index_find = data_1['Date'].index[data_1['Date'] == date][0]
            num_date = index_start - index_find
            prediction = model.predict(X_test[num_date][np.newaxis, :, :])
            prediction = scaler_price.inverse_transform(np.concatenate((prediction, np.zeros((len(prediction), 3))), axis=1))[:,0]
        elif np.array_equal(X_test, np.load('X_test_technical.npy')):
            index_start = data_2['Date'].index[data_2['Date'] == date_start_input2][0]
            index_find = data_2['Date'].index[data_2['Date'] == date][0]
            num_date = index_start - index_find
            prediction = model.predict(X_test[num_date][np.newaxis, :, :])
            prediction = scaler_price.inverse_transform(np.concatenate((prediction, np.zeros((len(prediction), 4))), axis=1))[:,0]
        return jsonify({'date': date, 'predicted_price': prediction[0]})
    elif model_name in models_stats:
        model, forecast = models_stats[model_name]
        fc = model.forecast(y = forecast, steps = 22)
        if np.array_equal(forecast, np.load('forecast_main.npy')):
            data_1_true = data_1.iloc[::-1]
            index_start = data_1_true['Date'].index[data_1_true['Date'] == date_start_var][0]
            index_find = data_1_true['Date'].index[data_1_true['Date'] == date][0]
            num_date = index_start - index_find
            data_1_true_true = data_1_true.drop(columns='Date')
            data_1_true_true = data_1_true_true.drop(columns='High')
            forecast_data = pd.DataFrame(fc, columns=data_1_true_true.columns)
            forecast_data = forecast_data.cumsum() + np.log(data_1_true_true).iloc[-23]
            forecast_data = np.exp(forecast_data)
        elif np.array_equal(forecast, np.load('forecast_tech.npy')):
            data_2_true = data_2.iloc[::-1]
            index_start = data_2_true['Date'].index[data_2_true['Date'] == date_start_var][0]
            index_find = data_2_true['Date'].index[data_2_true['Date'] == date][0]
            num_date = index_start - index_find
            data_2_true_true = data_2_true.drop(columns='Date')
            forecast_data = pd.DataFrame(fc, columns=data_2_true_true.columns)
            forecast_data['Close/Last'] = forecast_data['Close/Last'].cumsum() +data_2_true['Close/Last'].iloc[-23]
        elif np.array_equal(forecast, np.load('forecast_economic.npy')):
            data_3_true = data_3.iloc[::-1]
            index_start = data_3_true['Date'].index[data_3_true['Date'] == date_start_var][0]
            index_find = data_3_true['Date'].index[data_3_true['Date'] == date][0]
            num_date = index_start - index_find
            data_3_true_true = data_3_true.drop(columns='Date')
            forecast_data = pd.DataFrame(fc, columns=data_3_true_true.columns)
            forecast_data = forecast_data.cumsum() + data_3_true.iloc[-23]
        
        prediction = forecast_data['Close/Last'].iloc[num_date]
        return jsonify({'date': date, 'predicted_price': prediction})

    elif model_name in models_trans:
        model_path, scaler_price, X_test = models_trans[model_name]
        if np.array_equal(X_test, np.load('X_test_maintransformer.npy')): 
            model = TransformerModel(4, 2**6, 8, 3, 512, 0.1)
            
            model.load_state_dict(model_path)
            index_start = data_1['Date'].index[data_1['Date'] == date_start_input1][0]
            index_find = data_1['Date'].index[data_1['Date'] == date][0]
            num_date = index_start - index_find
            prediction = predict_stock_price(model, X_test, scaler_price)
        elif np.array_equal(X_test, np.load('X_test_techtransformer.npy')): 
            model = TransformerModel(5, 2**6, 8, 3, 512, 0.1)
            
            model.load_state_dict(model_path)
            index_start = data_2['Date'].index[data_2['Date'] == date_start_input2][0]
            index_find = data_2['Date'].index[data_2['Date'] == date][0]
            num_date = index_start - index_find
            prediction = predict_stock_price(model, X_test, scaler_price)
        elif np.array_equal(X_test, np.load('X_test_econtransformer.npy')): 
            model = TransformerModel(4, 2**6, 8, 3, 512, 0.1)
            
            model.load_state_dict(model_path)
            index_start = data_3['Date'].index[data_3['Date'] == date_start_input3][0]
            index_find = data_3['Date'].index[data_3['Date'] == date][0]
            num_date = index_start - index_find
            prediction = predict_stock_price(model, X_test, scaler_price)
        return jsonify({'date': date, 'predicted_price': prediction[num_date]})
if __name__ == '__main__':
    app.run(debug=True)
    app._static_folder = 'static'

