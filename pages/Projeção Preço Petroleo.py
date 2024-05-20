import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import timedelta
import plotly.express as px
import joblib
import os

# Load Data
# df = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', skiprows=1, thousands='.', decimal=',')[0]
df = pd.read_excel('data.xlsx')

# Data Preparation
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.set_index('Date')
df = df.loc['2020-01-01':]
df = df.sort_index()

# Data Transformation
scaler = MinMaxScaler()
scaler.fit(df)
scaled_train = scaler.transform(df)

# Define generator
n_input = 8
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# Define model
model_path = 'model.pkl'

if not os.path.exists(model_path):
    model = Sequential()
    model.add(LSTM(80, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.summary()
    # Treinando o modelo
    model.fit(generator, epochs=50)
    joblib.dump(model, model_path)
else:
    model = joblib.load(model_path)

# Gerando previsões
test_predictions = []
first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(30):
    current_pred = model.predict(current_batch)[0]
    test_predictions.append(current_pred)
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Transformando os valores de volta à escala original
true_predictions = scaler.inverse_transform(test_predictions)
df_predict = pd.DataFrame(true_predictions)

ultima_data = df.index.max()
proximas_datas = [ultima_data + timedelta(days=i) for i in range(1, 31)]
df_predict['data'] = proximas_datas

df_predict = df_predict.rename(columns={0: 'Predict'})
df_predict = df_predict.set_index('data')
df_predict.rename(columns={'data': 'Date', 'Predict': 'Price'}, inplace=True)
df['Type'] = 'Real'
df_predict['Type'] = 'Predicted'
df_combined = pd.concat([df, df_predict])
df_combined = df_combined.loc['2024-01-01':]

#fig_predict = px.line(df_predict, x='data', y='Predict', title='Previsão de preço dos Próximos 30 dias', line_shape='linear')
fig = px.line(df_combined, x=df_combined.index, y='Price', color='Type', title='Real vs Predicted Prices')
st.title('TECH 4 - ANÁLISE E PROJEÇÃO DO PREÇO DO PETRÓLEO')

st.plotly_chart(fig, use_container_width=True)
