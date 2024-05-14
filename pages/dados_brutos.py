import streamlit as st
import pandas as pd
import requests
import time


def mensagem_sucesso():
    sucesso = st.success('Arquivo baixado com sucesso!', icon = "✅")
    time.sleep(5)
    sucesso.empty()

st.title('TECH 4 - ANÁLISE E PROJEÇÃO DO PREÇO DO PETRÓLEO')
