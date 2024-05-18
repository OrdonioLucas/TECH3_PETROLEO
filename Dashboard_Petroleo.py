import streamlit as st
import pandas as pd
import plotly.express as px
import base64



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

#Funcão para formatar número
def formata_numero(valor,prefixo= ''):
    for unidade in ['','mil']:
        if valor <1000:
            return f'{prefixo}{valor:.2f}{unidade}'
        valor /=1000
    return f'{prefixo}{valor:.2f} milhões'

# Função para converter imagem em base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
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

aba1, aba2, aba3 = st.tabs(['Apresentação Geral','Análises', 'Predição'])

with aba1:
    
    # Texto introdutório
    st.markdown("""
    ### Análise de Variação do Petróleo

    Bem-vindo à análise de variação do preço do Petróleo do Grupo Tech 66, onde utilizamos a base do IPEA como fonte de dados para analisar a evolução do preço do Petróleo desde 1987. Nesta análise, iremos explorar os dados de forma geral, entendendo os motivos econômicos e de crise que levaram às grandes variações ao longo do tempo. Por último, forneceremos um modelo de machine learning capaz de prever os preços nos próximos 7 dias a partir do último dia disponibilizado na fonte de dados do site do IPEA.

    """)
    # Caminho da imagem
    image_path = 'petroleo_img.jpg'

    # Definindo a largura da imagem
    image_width = 850

    # HTML e CSS para centralizar a imagem
    html_code = f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpeg;base64,{image_to_base64(image_path)}" width="{image_width}">
        </div>
    """

    # Renderiza o HTML no Streamlit
    st.markdown(html_code, unsafe_allow_html=True)


    coluna1, coluna2, coluna3 = st.columns([1, 1, 1])

    with coluna1:
        st.metric('Preço Médio do Petróleo U$$', media_formatada)
        
    with coluna2:
        st.metric('Preço Mínimo do Petróleo U$$', min_formatada)

    with coluna3:
        st.metric('Preço Máximo do Petróleo U$$', max_formatada)

    st.plotly_chart(fig_preco_petroleo,use_container_width=True)

# Layout da aba "Análises"
with aba2:
    # Histograma dos Preços do Petróleo
    fig_hist = px.histogram(df_filtrado, x='Price', nbins=10, title='Distribuição dos Preços do Petróleo')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Boxplot dos Preços do Petróleo
    fig_box = px.box(df_filtrado, y='Price', title='Boxplot dos Preços do Petróleo')
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Gráfico de Barras dos Preços Médios por Ano
    df_avg = df_filtrado.groupby('Year', as_index=False)['Price'].mean()
    fig_bar = px.bar(df_avg, x='Year', y='Price', title='Preço Médio do Petróleo por Ano')
    st.plotly_chart(fig_bar, use_container_width=True)

# Layout da aba "Predição"
with aba3:
    st.write("Conteúdo de predição aqui")

