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

# funções auxiliares
def ml_error( model_name, y, yhat ):
    mae = mean_absolute_error( y, yhat )
    mape = mean_absolute_percentage_error( y, yhat )
    rmse = np.sqrt( mean_squared_error( y, yhat ) )
    
    return pd.DataFrame( { 'Model Name': model_name, 
                           'MAE': mae, 
                           'MAPE': mape,
                           'RMSE': rmse }, index=[0] )

#Load Data
#df = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', skiprows=1, thousands='.', decimal=',')[0]
df = pd.read_excel('data.xlsx')

# Data Preparation
#df = df.rename(columns={0:'Date',1:'Price'})
#df['Date'] = df['Date'].str.replace('/', '-')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
# Definindo a coluna 'A' como o índice
df = df.set_index('Date')
df = df.loc['2020-01-01':]

# Data Transformation
scaler = MinMaxScaler()
scaler.fit(df)
scaled_train = scaler.transform(df)

# define generator
n_input = 10
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

# define model
model = Sequential()
model.add(LSTM(80, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
#treinamento epochs
model.fit(generator,epochs=50)

# Predict
last_train_batch = scaled_train[-n_input:]
last_train_batch = last_train_batch.reshape((1, n_input, n_features))
model.predict(last_train_batch)


test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(7):
    
    # get the prediction value for the first batch
    current_pred = model.predict(current_batch)[0]
    
    # append the prediction into the array
    test_predictions.append(current_pred) 
    
    # use the prediction to update the batch and remove the first value
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
df_predict = pd.DataFrame(true_predictions)

# Verifique a última data no df1
ultima_data = df.index.max()

# Crie uma lista com os próximos 7 dias a partir da última data do df1
proximas_datas = [ultima_data + timedelta(days=i) for i in range(1, 8)]

# Adicione a lista de datas como uma nova coluna no df2
df_predict['data'] = proximas_datas

df_predict = df_predict.rename(columns={0:'Predict'})

fig_predict = px.line(df_predict, x='data', y='Predict', title='Previsão de preço dos Próximos 7 dias', line_shape='linear')

st.title('TECH 4 - ANÁLISE E PROJEÇÃO DO PREÇO DO PETRÓLEO')

st.plotly_chart(fig_predict,use_container_width=True)