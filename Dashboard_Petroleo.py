import streamlit as st
import pandas as pd
import plotly.express as px



# LOAD DATA
df = pd.read_html('http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view', skiprows=1, thousands='.', decimal=',')[0]

# TRATAMENTO DOS DADOS
df = df.rename(columns={0:'Date',1:'Price'})
df['Date'] = df['Date'].str.replace('/', '-')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

df = df.set_index('Date')

# Create additional features using the datetime index
df['Day_of_week'] = df.index.dayofweek
df['Month'] = df.index.month
df['Year'] = df.index.year
df['Year'] = df['Year'].astype(int)



st.set_page_config(layout = 'wide')

def formata_numero(valor,prefixo= ''):
    for unidade in ['','mil']:
        if valor <1000:
            return f'{prefixo}{valor:.2f}{unidade}'
        valor /=1000
    return f'{prefixo}{valor:.2f} milhões'

st.title('TECH 4 - ANÁLISE E PROJEÇÃO DO PREÇO DO PETRÓLEO')

st.sidebar.title('Filtros')
#ano = st.sidebar.selectbox('ANO', df['Year'].unique())



ano = st.sidebar.slider('Ano', 
    int(df['Year'].min()), 
    int(df['Year'].max()),
    (int(df['Year'].min()), int(df['Year'].max()))
)

df_filtrado = df[(df['Year'] >= ano[0]) & (df['Year'] <= ano[1])]

query_string = {'ano':ano}

# Criando variaveis dos cards

media =  df_filtrado['Price'].mean().round(0)
minimo = df_filtrado['Price'].min().round(0)
maximo = df_filtrado['Price'].max().round(0)

media_formatada = f'U${media:,.2f}'
min_formatada = f'U${minimo:,.2f}'
max_formatada = f'U${maximo:,.2f}'

fig_preco_petroleo = px.line(df_filtrado, x=df_filtrado.index, y='Price', title='Preço do Petróleo por Ano', line_shape='linear')

aba1, aba2, aba3 = st.tabs(['Preço Petróleo','Análises', 'Predição'])

coluna1, coluna2, coluna3 = st.columns([1, 1, 1])

with coluna1:
    st.metric('Preço Médio do Petróleo U$$', media_formatada)
    
with coluna2:
    st.metric('Preço Mínimo do Petróleo U$$', min_formatada)

with coluna3:
    st.metric('Preço Máximo do Petróleo U$$', max_formatada)

st.plotly_chart(fig_preco_petroleo,use_container_width=True)

