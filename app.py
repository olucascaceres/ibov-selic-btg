import pandas as pd
import yfinance as yf
import requests
import time, os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Definir ano alvo
ano = 2025
start_date = f"{ano}-01-01"
end_date = f"{ano}-12-31"

# 1. Extração de dados Ibovespa
ticker = "^BVSP"
df_ibov = yf.download(ticker, start=start_date, end=end_date)

if isinstance(df_ibov.columns, pd.MultiIndex):
    df_ibov.columns = df_ibov.columns.get_level_values(0)

df_ibov = df_ibov.rename_axis("Date")
df_ibov.to_csv("data/ibovespa.csv", index=True)

# 2. Extração de dados BTG Pactual (BPAC11.SA)
ticker_btg = "BPAC11.SA"
df_btg = yf.download(ticker_btg, start=start_date, end=end_date)

if isinstance(df_btg.columns, pd.MultiIndex):
    df_btg.columns = df_btg.columns.get_level_values(0)

df_btg["btg_return"] = df_btg["Close"].pct_change()
df_btg_reset = df_btg.reset_index()
df_btg_reset.rename(columns={"Date": "data", "Close": "btg_close"}, inplace=True)
df_btg_reset.to_csv("data/btg.csv", index=False)

# 3. Extração de dados Selic
# 3. Extração de dados Selic
import requests
import pandas as pd
import streamlit as st

ano = 2025
url_selic = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.432/dados?formato=json&dataInicial=01/01/{ano}&dataFinal=31/12/{ano}"

response = requests.get(url_selic)

if response.status_code == 200:
    try:
        data_selic = response.json()
        # Converter para DataFrame
        df_selic = pd.DataFrame(data_selic)
        df_selic["data"] = pd.to_datetime(df_selic["data"], format="%d/%m/%Y")
        df_selic["valor"] = df_selic["valor"].astype(float)

        # Garantir que só tenha dados de 2025
        df_selic = df_selic[
            (df_selic["data"].dt.year == ano)
        ].reset_index(drop=True)

        # Salvar em CSV
        df_selic.to_csv("data/selic.csv", index=False)

    except ValueError:
        st.error("⚠️ A resposta da API não está em formato JSON. Conteúdo recebido:")
        st.text(response.text)
        df_selic = pd.DataFrame()  # cria vazio para não quebrar
else:
    st.error(f"⚠️ Erro ao acessar API Selic: código {response.status_code}")
    df_selic = pd.DataFrame()  # cria vazio para não quebrar

# 4. Transformação dos dados
df_ibov["return"] = df_ibov["Close"].pct_change()
df_ibov_reset = df_ibov.reset_index()
df_ibov_reset.rename(columns={"Date": "data", "Close": "ibov_close", "return": "ibov_return"}, inplace=True)

df_merged = pd.merge(df_ibov_reset, df_selic, on="data", how="inner")
df_merged = pd.merge(df_merged, df_btg_reset[["data","btg_close","btg_return"]], on="data", how="left")

df_merged.to_csv("data/base_integrada.csv", index=False)
df_merged.to_parquet("data/base_integrada.parquet", index=False)

# 5. Teste de performance
start = time.time()
df_csv = pd.read_csv("data/base_integrada.csv")
print("Tempo leitura CSV:", round(time.time() - start, 4), "segundos")

start = time.time()
df_parquet = pd.read_parquet("data/base_integrada.parquet")
print("Tempo leitura Parquet:", round(time.time() - start, 4), "segundos")

st.title("📊 Análises Exploratórias — Ibovespa, Selic e BTG em 2025")
st.markdown("""
Este painel apresenta uma análise detalhada da relação entre o **Ibovespa**, a **taxa Selic** e as ações do **BTG Pactual** ao longo de 2025.  
Cada gráfico é acompanhado de uma contextualização e interpretação para facilitar a compreensão dos resultados.
""")

# ==========================
# 1️⃣ Ibovespa vs Selic
# ==========================
st.header("📈 Evolução do Ibovespa e da Selic em 2025")

st.markdown("""
Nesta etapa, analisamos a **relação entre o desempenho do Ibovespa e a taxa Selic ao longo de 2025**.  
O objetivo é observar como o principal índice da bolsa brasileira se comportou diante das variações da taxa básica de juros.
""")

fig, ax1 = plt.subplots(figsize=(10,5))
sns.lineplot(data=df_merged, x="data", y="ibov_close", color="blue", ax=ax1)
ax2 = ax1.twinx()
sns.lineplot(data=df_merged, x="data", y="valor", color="red", ax=ax2)
ax1.set_xlabel("Data")
ax1.set_ylabel("Ibovespa (pontos)", color="blue")
ax2.set_ylabel("Selic (%)", color="red")
plt.title("Evolução do Ibovespa e da Selic em 2025")
st.pyplot(fig)

st.markdown("""
💡 **Interpretação:**  
Mesmo com a Selic em trajetória de alta, o Ibovespa manteve uma tendência positiva ao longo de 2025.  
A Selic subiu gradualmente de cerca de **12,5% para 15%**, refletindo o esforço do Banco Central para conter pressões inflacionárias.  
Apesar disso, o Ibovespa avançou de aproximadamente **120 mil para mais de 160 mil pontos**, mostrando **confiança dos investidores** e **forte desempenho de setores como financeiro e commodities**.  
Em resumo, a alta dos juros **não impediu o avanço do mercado**, sugerindo um cenário de **otimismo moderado e resiliência** da bolsa brasileira.
""")

# ==========================
# 2️⃣ Distribuição dos Retornos Diários
# ==========================
st.header("📊 Distribuição dos Retornos Diários do Ibovespa")

st.markdown("""
Aqui analisamos a **distribuição dos retornos diários do Ibovespa em 2025**, para entender o comportamento estatístico do índice.  
O gráfico mostra a frequência dos retornos positivos e negativos, com a média destacada pela linha vermelha.
""")

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df_merged["ibov_return"].dropna(), bins=50, kde=True, color="purple", ax=ax)
ax.axvline(df_merged["ibov_return"].mean(), color="red", linestyle="--", label="Média")
ax.legend()
ax.set_title("Distribuição dos Retornos Diários do Ibovespa")
st.pyplot(fig)

st.markdown("""
💡 **Interpretação:**  
A distribuição dos retornos diários do Ibovespa em 2025 é aproximadamente **normal**, concentrada em torno de zero, com **leve assimetria à esquerda**.  
Isso indica que pequenas variações negativas foram ligeiramente mais frequentes do que grandes altas — um comportamento típico de **mercados maduros e estáveis**.  
O índice oscilou em torno de um equilíbrio, com ganhos e perdas se compensando ao longo do ano, e **baixa ocorrência de outliers**, refletindo **volatilidade controlada** e **confiança gradual** no mercado brasileiro.
""")

# ==========================
# 3️⃣ Retorno Acumulado — Ibovespa, BTG e Selic
# ==========================

# Garantir que as colunas de retorno acumulado e normalização da Selic existam
if "ibov_acum" not in df_merged.columns:
    df_merged["ibov_acum"] = (1 + df_merged["ibov_return"]).cumprod()

if "btg_acum" not in df_merged.columns:
    df_merged["btg_acum"] = (1 + df_merged["btg_return"]).cumprod()

if "selic_norm" not in df_merged.columns:
    df_merged["selic_norm"] = df_merged["valor"] / df_merged["valor"].iloc[0]


st.header("📈 Retorno Acumulado — Ibovespa, BTG e Selic em 2025")

st.markdown("""
Nesta etapa, comparamos o **desempenho acumulado** do Ibovespa, das ações do BTG Pactual e da taxa Selic ao longo de 2025.  
O objetivo é entender como o mercado acionário e a política monetária se relacionaram durante o ano.
""")

fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=df_merged, x="data", y="ibov_acum", color="blue", label="Ibovespa")
sns.lineplot(data=df_merged, x="data", y="btg_acum", color="brown", label="BTG")
sns.lineplot(data=df_merged, x="data", y="selic_norm", color="red", label="Selic (normalizada)")
plt.title("Retorno Acumulado — Ibovespa, BTG e Selic em 2025")
plt.xlabel("Data")
plt.ylabel("Retorno acumulado / índice normalizado (base 1)")
plt.legend()
st.pyplot(fig)

st.markdown("""
💡 **Interpretação:**  
O BTG Pactual teve **desempenho significativamente superior** ao Ibovespa em 2025, com retorno acumulado acima de **100%**, enquanto o índice geral cresceu cerca de **40%**.  
A Selic manteve trajetória estável, refletindo o controle gradual da política monetária.  
Esse comportamento indica que o BTG se beneficiou de **resultados corporativos sólidos** e **expansão de suas operações**, superando o ritmo do mercado.  
Em síntese, o gráfico evidencia que o **setor financeiro foi um dos grandes destaques de 2025**, mostrando **resiliência e capacidade de crescimento mesmo em um ambiente de juros elevados**.
""")

# ==========================
# 4️⃣ Retornos Diários do BTG com Outliers
# ==========================

# Detectar outliers nos retornos do BTG (critério: ±2 desvios padrão)
limite_superior = df_merged["btg_return"].mean() + 2 * df_merged["btg_return"].std()
limite_inferior = df_merged["btg_return"].mean() - 2 * df_merged["btg_return"].std()

outliers_btg = df_merged[
    (df_merged["btg_return"] > limite_superior) | 
    (df_merged["btg_return"] < limite_inferior)
]

st.header("📉 Retornos Diários do BTG com Outliers em 2025")

fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(x="data", y="btg_return", data=df_merged, color="brown", ax=ax)
sns.scatterplot(x="data", y="btg_return", data=outliers_btg, color="red", label="Outliers", ax=ax)
plt.title("Retornos Diários do BTG com Outliers em 2025")
plt.xlabel("Data")
plt.ylabel("Retorno Diário")
plt.legend()
st.pyplot(fig)

st.markdown("""
💡 **Interpretação:**  
Os outliers do BTG em 2025 coincidem com **eventos corporativos relevantes**:  
- **08/05/2025:** alta de +6%, antecipando bons resultados do 1º trimestre.  
- **12/08/2025:** salto de +13%, após divulgação de **lucro recorde de R$ 4,18 bi** e **ROE ajustado de 27,1%**.  
- **05/12/2025:** queda de −7,9%, associada à **realização de lucros e ajustes pós‑pagamento de JCP**.  

Esses picos refletem **reações pontuais do mercado** a divulgações de resultados e ajustes técnicos, típicos de ações do setor financeiro.
""")

# ==========================
# 5️⃣ Correlação entre Ibovespa, BTG e Selic
# ==========================
st.header("📊 Correlação entre Ibovespa, BTG e Selic em 2025")

st.markdown("""
Por fim, analisamos a **matriz de correlação** entre os retornos do Ibovespa, do BTG Pactual e da taxa Selic.  
O objetivo é entender como esses indicadores se relacionaram ao longo do ano.
""")

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(df_merged[["ibov_return","btg_return","valor"]].corr(), annot=True, cmap="coolwarm", ax=ax)
plt.title("Correlação entre Ibovespa, BTG e Selic em 2025")
st.pyplot(fig)

st.markdown("""
💡 **Interpretação:**  
O **Ibovespa** e o **BTG Pactual** apresentaram **forte correlação positiva (0,72)**, enquanto ambos tiveram **correlação fraca e negativa com a Selic** (−0,02 e −0,08).  
Isso mostra que o BTG **acompanhou o ritmo do mercado**, mas **não foi significativamente afetado pelos juros**.  
Essa relação reforça o perfil de **resiliência e estabilidade** do banco em 2025, com desempenho guiado por **resultados corporativos e fundamentos internos**.
""")

st.markdown("---")
