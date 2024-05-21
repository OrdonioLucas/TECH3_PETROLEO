import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from streamlit_option_menu import option_menu
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
import time
#-------

#-- Config da página
st.set_page_config(page_title="Tech4 - Petróleo",
                   layout='wide',
                   page_icon="chart_with_upwards_trend")

# Função para carregar dados e armazenar no cache do navegador
@st.cache_data
def carregar_dados():
    df = pd.read_excel('data.xlsx')
    # TRATAMENTO DOS DADOS
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df = df.set_index('Date')
    return df

@st.cache_data
def carregar_dados_2():
    df2 = df
    # TRATAMENTO DOS DADOS
    df2 = df2.loc['2020-01-01':]
    df2 = df2.sort_index()
    return df2

# Carregando os dados principal
df = carregar_dados()

# Carregando os dados pro modelo preditivo
df2 = carregar_dados_2()

# Criar recursos adicionais usando o índice datetime
df['Day_of_week'] = df.index.dayofweek
df['Month'] = df.index.month
df['Year'] = df.index.year
df['Year'] = df['Year'].astype(int)

# Função para formatar número
def formata_numero(valor, prefixo=''):
    for unidade in ['','mil']:
        if valor < 1000:
            return f'{prefixo}{valor:.2f}{unidade}'
        valor /= 1000
    return f'{prefixo}{valor:.2f} milhões'

# Função para converter imagem em base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Barra Lateral - Índice de Navegação
with st.container():
    with st.sidebar:
        pagina = option_menu('Ir para:', ['Análises dos Dados', 'Apresentação', 'Análise Macro','Predição'], icons=['activity', 'house', 'calendar2-week','graph-up-arrow'], menu_icon="cast", default_index=1)
    pagina

    # Filtros de Ano na Barra Lateral
    ano = st.sidebar.slider('Filtro de Datas', 
        int(df['Year'].min()), 
        int(df['Year'].max()),
        (int(df['Year'].min()), int(df['Year'].max()))
    )

    st.sidebar.success("☝️ Selecione um tópico acima.")
    df_filtrado = df[(df['Year'] >= ano[0]) & (df['Year'] <= ano[1])]

# Criando variáveis dos cards
media = df_filtrado['Price'].mean().round(0)
minimo = df_filtrado['Price'].min().round(0)
maximo = df_filtrado['Price'].max().round(0)

media_formatada = f'U${media:,.2f}'
min_formatada = f'U${minimo:,.2f}'
max_formatada = f'U${maximo:,.2f}'

# Gráfico de Preço do Petróleo
fig_preco_petroleo = px.line(df_filtrado, x=df_filtrado.index, y='Price', title='Preço do Petróleo por Ano', line_shape='linear')

#Gráfico de preço petróleo específico
def plot_data(start_date, end_date):
    datas = df[(df['Year'] >= start_date) & (df['Year'] <= end_date)]

    fig_espec = px.line(datas, x= datas.index, y='Price', title= f'Barril de petróleo em US$ nas datas entre {start_date} e {end_date}', line_shape='linear')
    st.plotly_chart(fig_espec)


# Conteúdo Principal baseado na seleção do Índice de Navegação
if pagina == 'Apresentação':
    st.title('TECH 4 - ANÁLISE E PROJEÇÃO DO PREÇO DO PETRÓLEO')
    st.markdown("""
                ### Análise de Variação do Petróleo

                Bem-vindo à análise de variação do preço do Petróleo do Grupo Tech 66 🎉, onde utilizamos a base do IPEA como fonte de dados para analisar a evolução do preço do Petróleo desde 1987. Nesta análise, iremos explorar os dados de forma geral, entendendo os motivos econômicos e de crise que levaram às grandes variações ao longo do tempo. Por último, forneceremos um modelo de machine learning capaz de prever os preços nos próximos 7 dias a partir do último dia disponibilizado na fonte de dados do site do IPEA.

                """)
    image_path = 'petroleo_img.jpg'
    image_width = 850
    html_code = f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/jpeg;base64,{image_to_base64(image_path)}" width="{image_width}">
        </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)
    with st.container():
        st.subheader('Grupo 66 - Autores:')
        st.write('''
                 - Flademir de Albuquerque
                
                 - Lucas Ordonio
                
                 - Francisco das Chagas Peres Júnior
                 
                 ---
                 ''')
    with st.container():
        st.subheader('Descrição:')
        st.markdown('''
                    Para a resolução do desafio iniciamos utilizando web scraping para entendermos todo o processo, disponíel no notebook [1](https://github.com/OrdonioLucas/TECH3_PETROLEO/blob/main/Dashboard_Petroleo.py).
                    
                    Fora utilizada as seguintes técnicas e tecnologias:
                    * LSTM para a resolução do problema de Time Series
                    * Streamlit com o streamlit_option_menu para controle do menu.
                    
                    ---
                    ### Contribuições

                    Contribuições são sempre bem-vindas. Para contribuir:

                    1 - Faça um fork do repositório.

                    2 - Crie uma branch para sua feature (git checkout -b feature/NovaFeature).

                    3 - Faça commit de suas mudanças (git commit -am 'Adicionando uma nova feature').

                    4 - Faça push para a branch (git push origin feature/NovaFeature).

                    5 - Abra um Pull Request.

                    ---
                    Link do GitHub:
                    * https://github.com/OrdonioLucas/TECH3_PETROLEO/tree/main
                    ''')

elif pagina == 'Análises dos Dados':
    st.title('Análises dos Dados')
    with st.container():
        coluna1, coluna2, coluna3 = st.columns([1, 1, 1])
        with coluna1:
            st.metric('Preço Médio do Petróleo U$$', media_formatada)
        with coluna2:
            st.metric('Preço Mínimo do Petróleo U$$', min_formatada)
        with coluna3:
            st.metric('Preço Máximo do Petróleo U$$', max_formatada)
        st.write('')
        st.markdown('''
                    Os valores do [site do ipea](http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view) representam bem a situação atual do valor do preço do petróleo,
                    demonstram uma proeminencia de valores inconstantes, mostrando a sua natureza volátil no mercado.

                    O valor na figura 1(preços médios, mínimos e máximos) demonstra a capacidade de aumentar abruptamente. Entretanto ao visualizarmos o gráfico boxplot e a distribuição de preços
                    abaixo é possível perceber que a maior parte dos dados está distribuída abaixo de 80 dólares
                    ''')
    fig_hist = px.histogram(df_filtrado, x='Price', nbins=10, title='Distribuição dos Preços do Petróleo')
    st.plotly_chart(fig_hist, use_container_width=True)
    fig_box = px.box(df_filtrado, y='Price', title='Boxplot dos Preços do Petróleo')
    st.plotly_chart(fig_box, use_container_width=True)
    df_avg = df_filtrado.groupby('Year', as_index=False)['Price'].mean()
    with st.container():
        st.markdown('''
                    O foco não deve ser na distribuição do valor, a base de dados começa em 1987, onde os valores eram consideravalemnte menores
                    devido a uma gigantesca série de fatores(inflação, mudanças economicas) que não entraremos no assunto. 
                    
                    O ponto é, o valor tem crescido em média, sendo afetado por combinações de oferta, demandas, políticas internacionais e eventos inesperados como pandemias e guerras​.
                    O valor do preço do barril do petróleo tem mantido uma constancia interessante(figura abaixo), um produto vital, mesmo possuindo finita quantidade.
                    ''')        
        fig_bar = px.bar(df_avg, x='Year', y='Price', title='Preço Médio do Petróleo por Ano')
        st.plotly_chart(fig_bar, use_container_width=True)

elif pagina == 'Análise Macro':
    st.title('Análise Macro')
    with st.container():
        st.subheader('Introdução')
        st.write("""
                Tensões geopolíticas, decisões da OPEP, o mercado mundial e avanços na exploração de energia são os principais fatores para as variações no preço do petróleo. Como principal fonte de energia, as variações no seu preço impactam profundamente a economia global, influenciando custos de produção, preços dos bens de consumo e a inflação.
             """)
        st.write("""
                Além de afetar mercados financeiros e políticas energéticas, as mudanças no preço do petróleo têm implicações sociais e ambientais. A volatilidade pode desencadear crises em países dependentes do petróleo e aumentar os custos de energia para consumidores. Também pode acelerar a transição para fontes de energia renováveis, enquanto governos e empresas buscam reduzir a dependência do petróleo e mitigar mudanças climáticas. Analisar esses fatores é crucial para entender o panorama econômico atual e futuro.
             """)
        st.plotly_chart(fig_preco_petroleo, use_container_width=True)
        st.subheader("Contexto Histórico:")
        st.write("""
                 Em 1960, foi criada a Organização dos Países Exportadores de Petróleo (OPEP) com o objetivo de controlar a exploração do petróleo nos principais países produtores da época: Irã, Iraque, Kuwait, Arábia Saudita e Venezuela. A formação da OPEP fortaleceu esses países, permitindo-lhes negociar acordos mais favoráveis com as companhias estrangeiras e, consequentemente, influenciar o preço do petróleo no mercado global.
                 """)
        st.write("""
                 Na década de 1970, com a expansão da OPEP, a nacionalização do petróleo se tornou crescente. Com maior controle sobre a produção e as decisões de preços, os membros da OPEP elevaram os preços do petróleo, o que resultou na crise de 1973. Crise que influenciou ao incentivo ao uso do Etanol e mistura de etanol na gasolina no Brasil.
                 """)
    with st.container():
        st.subheader('Análise:')        
        st.write("""
                Apesar invasão do Kuwait em 1990 e da guerra do Iraque em 2003, a produção de óleo não foi afetada, mantendo os preços regulados visto que a Arábia Saudita e outros membros da OPEP não envolvidos no conflito, conseguiram manter o fornecimento de óleo mundial.
                """)
        plot_data(1990, 2003)

    with st.container():        
        st.write("""
                Em 2008, houve um aumento significativo no preço do petróleo, devido à pressão exercida pelos membros mais pobres da OPEP sobre a Arábia Saudita para elevar os preços. Isso resultou em uma alta histórica, com o barril ultrapassando os US$140. No entanto, com a crise financeira global de 2008, a demanda por petróleo caiu drasticamente, levando a uma queda acentuada nos preços.
                 """)
        plot_data(2008, 2012)
    
    with st.container():        
        st.write("""
                 Durante 2014 e 2015 houve o Oil Shock, onde os membros da OPEP frequentemente ultrapassaram seu teto de produção, ao mesmo tempo em que a economia chinesa desacelerava. Nos Estados Unidos, a produção de petróleo praticamente dobrou desde 2008, aproximando o país da independência energética. Esses fatores contribuíram para um grande colapso nos preços do petróleo, que se estendeu até o início de 2016.                 
                 """)
        plot_data(2014, 2015)

    with st.container():        
        st.write("""
                De 2018 a 2020, o mercado de petróleo experimentou várias altas e baixas devido à entrada de novos concorrentes da OPEP, como o Brasil com o pré-sal, o shale gas e o tight oil dos Estados Unidos, além do óleo das areias betuminosas do Canadá. Esses novos produtores forçaram os países membros da OPEP a reduzir significativamente sua produção entre 2017 e 2018 para manter os preços elevados.
                
                Em 2020, devido à pandemia, a demanda por petróleo caiu drasticamente, resultando no preço mais baixo desde dezembro de 2018 (US$ 9,12 por barril). Em resposta, a OPEP+ (composta pelos membros da OPEP e mais 10 países convidados) decidiu cortar a produção em mais de 9 milhões de barris por dia para elevar os preços.
                
                 """)
        plot_data(2018, 2020)

    with st.container():        
        st.write("""
                Devido à invasão da Ucrânia em 2022 e à interrupção do fornecimento de petróleo e gás natural para a Europa, o preço do petróleo disparou, atingindo quase a alta histórica de 133 dólares em março de 2022. No entanto, em março de 2023, com o estabelecimento de novas rotas de fornecimento de gás e petróleo, o preço já havia caído para 71 dólares o barril.
                 """)
        plot_data(2022, 2024)

    with st.container():
        st.subheader('Conclusão:')
        st.markdown('''
                    De acordo com o último World Energy Outlook (2023), a projeção é de que a demanda por combustíveis fósseis diminuirá, com base no Cenário de Políticas Públicas Declaradas (STEPS, em inglês). Para alguns combustíveis, o pico de demanda já foi atingido. Como consequência, tecnologias baseadas em combustíveis fósseis estão perdendo mercado para tecnologias de energia limpa em diversos setores, e certas tecnologias de combustíveis fósseis já atingiram seu pico de vendas.

                    Nas duas últimas décadas a demanda de petróleo subiu 18 milhões de barris por dia, muito devido ao aumento do transporte rodoviário. No mesmo período a atividade rodoviária aumentou quase 65% e a frota mundial de carros expandiu mais de 600 milhões de veículos. Atualmente o transporte rodoviário é responsável por 45% da demanda de petróleo.

                    O crescimento astronômico do carro elétrico tem impactado severamente a demanda por óleo no setor de transporte. Em 2020 o carro elétrico era responsável por somente 4% das vendas, em 2023 passou a ser 18% com projeções de aumentar ainda mais nos próximos anos. As vendas de ônibus com motor à combustão tiveram o seu pico em 2020, sendo substituído cada vez mais por ônibus elétricos. Sendo assim, o setor de transporte rodoviário tem uma demanda por petróleo em declínio.

                    Apesar da demanda de petróleo e seus derivados para a indústria petroquímica, aviação e transporte hidroviário continue a crescer, conforme a projeção do STEPS, até 2050, não será o suficiente para combater as reduções de demanda para os setores de transporte rodoviário, energia e setores de construção. Sendo assim, a projeção é que o pico da demanda de petróleo ocorra antes de 2030, contudo a queda será lenta.

                    Regionalmente, países desenvolvidos tiveram seu pico de demanda por petróleo em 2005, nos próximos anos a demanda de petróleo na China irá enfraquecer e começar seu declínio. Porém em outros países emergentes que tem demanda por carros e consequentemente petróleo crescente e continua a crescer até 2050.

                    ---

                    ### Referencias:

                    * https://www.eia.gov/totalenergy/data/browser/#/?f=A&start=1949&end=2023&charted=4-6-7-14

                    * https://ourworldindata.org/fossil-fuels

                    * https://iea.blob.core.windows.net/assets/86ede39e-4436-42d7-ba2a-edf61467e070/WorldEnergyOutlook2023.pdf

                    * https://scholarworks.gsu.edu/cgi/viewcontent.cgi?article=1089&context=political_science_facpub 

                    * https://diplomatique.org.br/a-nova-geopolitica-do-petroleo-no-seculo-xxi/ 

                    * https://en.wikipedia.org/wiki/Petroleum 

                    * https://en.wikipedia.org/wiki/OPEC 

                    * https://www.cirsd.org/en/horizons/horizons-spring-2015--issue-no3/oil-shock-—-decoding-the-causes-and-consequences-of-the-2014-oil-price-drop

                    ''')


elif pagina =='Predição':
    # Data Transformation
    scaler = MinMaxScaler()
    scaler.fit(df2)
    scaled_train = scaler.transform(df2)

    # Define generator
    n_input = 8
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

    # Define model
    model_path = 'model.pkl'

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    last_rows = np.random.randn(1, 1)

    for i in range(1, 101):
        new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
        status_text.text("%i%% Complete" % i)
        progress_bar.progress(i)
        last_rows = new_rows
        time.sleep(0.05)

    progress_bar.empty()

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
    st.subheader('Considerações Finais:')
    st.markdown('''
                O setor energético como todo tem sofrido nos últimos anos, primeiramente pela pandemia e seguido pela invasão da Ucrânia, impactando consumidores e produtores de energia pela volatilidade nos preços.

                Embora alguns indicadores reflitam que estamos retomando índices de consumo e preços pré-pandemia, não há indicadores suficientes para que a crise energética de 2020-2023 tenha passado. A guerra na Ucrânia, a instabilidade no Oriente Médio e os indicativos claros para a transição energética são sinais de que ainda estamos em um período instável.

                Dado o período de instabilidade e a manutenção das políticas de corte de produção por parte dos membros da OPEP+, há uma tendência dos preços do barril de petróleo à subir e continuar nesse ritmo até que a demanda volte a descer (como visto constantemente no gráfico histórico de preços).
                ''')