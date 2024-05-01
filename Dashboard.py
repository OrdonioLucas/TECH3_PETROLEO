import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(layout = 'wide')

def formata_numero(valor,prefixo= ''):
    for unidade in ['','mil']:
        if valor <1000:
            return f'{prefixo}{valor:.2f}{unidade}'
        valor /=1000
    return f'{prefixo}{valor:.2f} milhões'

st.title('DASHBOARD DE VENDAS ODONIO SHOP "🙅‍♂️" :shopping_trolley:')

url = 'https://labdados.com/produtos'
regioes = ['Brasil', 'Centro-Oeste', 'Nordeste', 'Norte', 'Sudeste', 'Sul']

st.sidebar.title('Filtros')
regiao = st.sidebar.selectbox('Região', regioes)

if regiao == 'Brasil':
    regiao = ''

todos_anos = st.sidebar.checkbox('Dados de todo o período', value = True)

if todos_anos:
    ano = ''
else:
    ano = st.sidebar.slider('Ano', 2020, 2023,(2020,2023))

query_string = {'regiao':regiao.lower(), 'ano':ano}

response = requests.get(url, params= query_string)
dados = pd.DataFrame.from_dict(response.json())
dados['Data da Compra'] = pd.to_datetime(dados['Data da Compra'], format = '%d/%m/%Y')


filtro_vendedores = st.sidebar.multiselect('Vendedores', dados['Vendedor'].unique())

if filtro_vendedores:
    dados = dados[dados['Vendedor'].isin(filtro_vendedores)]

## Construindo tabelas auxiliares
### Tabelas Receita
receita_estado = dados.groupby('Local da compra')[['Preço']].sum()
receita_estado = dados.drop_duplicates(subset='Local da compra')[['Local da compra','lat','lon']].merge(receita_estado,left_on='Local da compra',right_index=True).sort_values('Preço',ascending=False)
receita_mensal = dados.set_index('Data da Compra').groupby(pd.Grouper(freq='M'))['Preço'].sum().reset_index()
receita_mensal['Ano'] = receita_mensal['Data da Compra'].dt.year
receita_mensal['Mes'] = receita_mensal['Data da Compra'].dt.month_name()
receita_categorias = dados.groupby('Categoria do Produto')[['Preço']].sum().sort_values('Preço',ascending=False)

### Tabela Vendas

qtd_vendas_estado = dados.groupby('Local da compra')[['Local da compra']].count()
qtd_vendas_estado = dados.drop_duplicates(subset='Local da compra')[['Local da compra','lat','lon']].merge(qtd_vendas_estado,left_on='Local da compra',right_index=True).sort_values('Local da compra',ascending=False)
qtd_vendas_mensal  = dados.set_index('Data da Compra').groupby(pd.Grouper(freq='M'))['Local da compra'].count().reset_index()
qtd_vendas_mensal['Ano'] = qtd_vendas_mensal['Data da Compra'].dt.year
qtd_vendas_mensal['Mes'] = qtd_vendas_mensal['Data da Compra'].dt.month_name()
qtd_vendas_mensal = qtd_vendas_mensal.rename(columns= {'Local da compra': 'Qtd de vendas'})
qtd_vendas_categorias = dados['Categoria do Produto'].value_counts().reset_index()
qtd_vendas_categorias.columns = ['Categoria do Produto', 'Quantidade de Vendas']

###Tabelas Vendedores 
vendedores = pd.DataFrame(dados.groupby('Vendedor')['Preço'].agg(['sum', 'count']))

## Gráficos receita Mensal

fig_mapa_receita = px.scatter_geo(receita_estado,
                                  lat='lat',
                                  lon='lon',
                                  scope='south america',
                                  size='Preço',
                                  template= 'seaborn',
                                  hover_name='Local da compra',
                                  hover_data={'lat':False,'lon':False},
                                  title= 'Receita por Estado')


fig_receita_mensal = px.line(receita_mensal,
                                                        x = 'Mes',
                                                        y = 'Preço',
                                                        markers = True,
                                                        range_y = (0, receita_mensal.max()),
                                                        color='Ano',
                                                        line_dash = 'Ano',
                                                        title = 'Receita mensal')

fig_receita_mensal.update_layout(yaxis_title = 'Receita')


fig_receita_estado = px.bar(receita_estado.head(),
                            x = 'Local da compra',
                            y = 'Preço',
                            text_auto= True,
                            title = 'Top estados (receita)')

fig_receita_estado.update_layout(yaxis_title = 'Receita')

fig_receita_categorias = px.bar(receita_categorias.head(),
                            text_auto= True,
                            title = 'Receita por categoria')

## gráficos página quantidade de vendas
fig_mapa_qtd_vendas = px.scatter_geo(qtd_vendas_estado,
                                  lat='lat',
                                  lon='lon',
                                  scope='south america',
                                  size='Local da compra_y',
                                  template= 'seaborn',
                                  hover_name='Local da compra',
                                  hover_data={'lat':False,'lon':False},
                                  title= 'Receita por Estado')

fig_qtd_vendas_mensal = px.line(qtd_vendas_mensal,
                                                        x = 'Mes',
                                                        y = 'Qtd de vendas',
                                                        markers = True,
                                                        range_y = (0, qtd_vendas_mensal.max()),
                                                        color='Ano',
                                                        line_dash = 'Ano',
                                                        title = 'Qtd de vendas mensal')

fig_vendas_estado = px.bar(qtd_vendas_estado.head(),
                            x = 'Local da compra',
                            y = 'Local da compra_y',
                            text_auto= True,
                            title = 'Top estados (vendas)')

fig_vendas_categorias = px.bar(qtd_vendas_categorias.head(),
                            x= 'Categoria do Produto',
                            y= 'Quantidade de Vendas',
                            text_auto= True,
                            title = 'Qtd vendas por categoria')

## Visualização no Streamlit

## Definindo a quantidade de abas / páginas a serem exibidas no painel do streamlit

aba1, aba2, aba3 = st.tabs(['Receita','Quantidade de Vendas', 'Vendedores'])

with aba1:
    coluna1,coluna2 = st.columns(2)
    with coluna1:
        st.metric('Receita',formata_numero(dados['Preço'].sum().round(0),'R$'))
        st.plotly_chart(fig_mapa_receita,use_container_width = True)
        st.plotly_chart(fig_receita_estado,use_container_width=True)
    with coluna2:
        st.metric('Quantidade de vendas',formata_numero(dados.shape[0]))
        st.plotly_chart(fig_receita_mensal,use_container_width = True)
        st.plotly_chart(fig_vendas_categorias,use_container_width=True)

with aba2:
    coluna1,coluna2,coluna3 = st.columns([2, 2, 1])
    with coluna1:
        st.metric('Receita',formata_numero(dados['Preço'].sum().round(0),'R$'))
        st.plotly_chart(fig_mapa_qtd_vendas,use_container_width = True)
        st.plotly_chart(fig_vendas_estado,use_container_width = True)
    with coluna2:
        st.metric('Quantidade de vendas',formata_numero(dados.shape[0]))
        st.plotly_chart(fig_qtd_vendas_mensal,use_container_width = True)
        st.plotly_chart(fig_vendas_categorias,use_container_width = True)

    with coluna3:
        st.metric('Quantidade de estados',dados['Local da compra'].nunique())

        
with aba3:
    qtd_vendedores = st.number_input('Quantidade de vendedores', 2, 10, 5)
    coluna1,coluna2 = st.columns(2)
    with coluna1:
        st.metric('Receita',formata_numero(dados['Preço'].sum().round(0),'R$'))
        fig_receita_vendedores = px.bar(
        vendedores[['sum']].sort_values('sum', ascending=False).head(qtd_vendedores),
        x='sum',
        y=vendedores[['sum']].sort_values(['sum'], ascending=False).head(qtd_vendedores).index,
        text_auto=True,
        title=f'Top {qtd_vendedores} vendedores (receita)'
        )
        st.plotly_chart(fig_receita_vendedores,use_container_width=True)

    with coluna2:
        st.metric('Quantidade de vendas',formata_numero(dados.shape[0]))
        fig_venda_vendedores = px.bar(
        vendedores[['count']].sort_values('count', ascending=False).head(qtd_vendedores),
        x='count',
        y=vendedores[['count']].sort_values(['count'], ascending=False).head(qtd_vendedores).index,
        text_auto=True,
        title=f'Top {qtd_vendedores} vendedores (Quantidade de vendas)'
        )
        st.plotly_chart(fig_venda_vendedores,use_container_width=True)
    

#código para exibir o df  no streamlit
# st.dataframe(dados)

