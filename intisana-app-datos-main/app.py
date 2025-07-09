import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import folium
from streamlit_folium import st_folium
from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import requests

# Configuración inicial
st.set_page_config(page_title="Comparador Solar Mica", layout="wide")
st.title("☀️ Comparador de Radiación Solar y Humedad - Mica, Ecuador")

# Blog / hipótesis
with st.expander("📘 Hipótesis del Estudio", expanded=True):
    st.markdown("""
    > **Hipótesis:** Días con niveles de radiación solar superiores a 630 W/m² pueden acelerar la evaporación del agua y contribuir al desabastecimiento.  
    Compararemos datos históricos con registros actuales (NASA POWER) en la zona de la represa Mica, Ecuador.
    """)

# 🌍 Mapa ubicación
with st.expander("🗺️ Ubicación del Estudio", expanded=False):
    m = folium.Map(location=[-0.35, -78.35], zoom_start=10)
    folium.Marker(
        location=[-0.35, -78.35],
        popup="Represa Mica",
        tooltip="Mica - Ecuador",
        icon=folium.Icon(color="green"),
    ).add_to(m)
    st_folium(m, width=700)

# Subida de archivo
archivo = st.file_uploader("📤 Sube archivo .xlsx con datos históricos (radiación y humedad)", type=["xlsx"])

if "historial" not in st.session_state:
    st.session_state.historial = []

if archivo:
    try:
        df_hist = pd.read_excel(archivo, header=11)
        df_hist.columns = df_hist.columns.str.strip()
        df_hist = df_hist[['Fecha', 'Valor']].rename(columns={'Valor': 'Radiacion'})
        df_hist['Fecha'] = pd.to_datetime(df_hist['Fecha'])
        df_hist['Periodo'] = 'Histórico'
        df_hist['Humedad'] = None

        # 🗓️ Filtro por fecha
        fecha_min, fecha_max = df_hist['Fecha'].min(), df_hist['Fecha'].max()
        rango = st.date_input("🗓️ Filtrar fechas históricas", [fecha_min, fecha_max],
                              min_value=fecha_min, max_value=fecha_max)
        df_hist = df_hist[(df_hist['Fecha'] >= pd.to_datetime(rango[0])) & (df_hist['Fecha'] <= pd.to_datetime(rango[1]))]

        # 🌐 Datos de NASA
        st.info("Cargando datos recientes desde la NASA POWER...")
        lat, lon = -0.22, -78.36
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,RH2M&community=AG&longitude={lon}&latitude={lat}&start=20240101&end=20250622&format=JSON"
        nasa = requests.get(url).json()

        df_nasa = pd.DataFrame({
            'Fecha': list(nasa['properties']['parameter']['ALLSKY_SFC_SW_DWN'].keys()),
            'Radiacion': list(nasa['properties']['parameter']['ALLSKY_SFC_SW_DWN'].values()),
            'Humedad': list(nasa['properties']['parameter']['RH2M'].values())
        })
        df_nasa['Fecha'] = pd.to_datetime(df_nasa['Fecha'])
        df_nasa['Periodo'] = 'NASA 2024–2025'

        # Combinar datos
        df = pd.concat([df_hist, df_nasa]).reset_index(drop=True)
        df['Radiacion_Alta'] = df['Radiacion'] > 630

        # Guardar en historial de comparaciones
        st.session_state.historial.append(df.copy())

        # 📈 Gráfica de radiación
        st.subheader("📊 Boxplot de Radiación Solar")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='Periodo', y='Radiacion', palette=['orange', 'darkorange'], ax=ax)
        plt.axhline(630, color='red', linestyle='--', label='Umbral 630 W/m²')
        plt.legend()
        st.pyplot(fig)

        # Porcentaje días críticos
        st.subheader("🔴 Porcentaje de días con radiación > 630 W/m²")
        porcentajes = df.groupby('Periodo')['Radiacion_Alta'].mean() * 100
        st.bar_chart(porcentajes)

        # 📥 Descargar resultados
        st.subheader("📥 Descargar comparación como CSV")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Descargar CSV", csv, f"comparacion_{datetime.now().date()}.csv", "text/csv")

    except Exception as e:
        st.error(f"⚠️ Error al procesar el archivo: {e}")

# 🗃️ Mostrar historial de comparaciones anteriores
if st.session_state.historial:
    with st.expander("🗃️ Comparaciones anteriores (esta sesión)", expanded=False):
        for i, comp in enumerate(st.session_state.historial):
            st.markdown(f"**Comparación #{i+1}** – {comp['Fecha'].min().date()} a {comp['Fecha'].max().date()}")
            st.dataframe(comp[['Fecha', 'Radiacion', 'Humedad', 'Periodo']].tail(5))

# app.py



st.set_page_config(page_title="☀️ Comparador Solar Mica", layout="wide")
st.title("☀️ Comparador Radiación Solar y Humedad - Mica, Ecuador")

# 📂 Cargar archivos locales
@st.cache_data
def cargar_datos_locales():
    df_radiacion = pd.read_excel("C09-Mica_Campamento_Radiación_solar-Diario.xlsx", header=11)
    df_presion = pd.read_excel("C09-Mica_Campamento_Presion_atmosférica-Diario.xlsx", header=11)
    df_humedad = pd.read_excel("C09-Mica_Campamento_Humedad_relativa-Diario.xlsx", header=11)

    df_radiacion = df_radiacion[['Fecha', 'Valor']].rename(columns={'Valor': 'Radiacion'})
    df_presion = df_presion[['Fecha', 'Valor']].rename(columns={'Valor': 'Presion'})
    df_humedad = df_humedad[['Fecha', 'Valor']].rename(columns={'Valor': 'Humedad'})

    for df in [df_radiacion, df_presion, df_humedad]:
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df.dropna(subset=['Fecha'], inplace=True)

    df_total = df_radiacion.merge(df_presion, on='Fecha').merge(df_humedad, on='Fecha')
    df_total['Radiacion_Alta'] = df_total['Radiacion'] > 630
    return df_total

# 🌐 Cargar datos NASA POWER API
@st.cache_data
def cargar_datos_nasa():
    latitude = -0.22
    longitude = -78.36
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,RH2M&community=AG&longitude={longitude}&latitude={latitude}&start=20240101&end=20250622&format=JSON"
    data = requests.get(url).json()['properties']['parameter']
    df_nasa = pd.DataFrame({
        'Fecha': pd.to_datetime(list(data['ALLSKY_SFC_SW_DWN'].keys())),
        'Radiacion': list(data['ALLSKY_SFC_SW_DWN'].values()),
        'Humedad': list(data['RH2M'].values())
    })
    df_nasa['Radiacion_Alta'] = df_nasa['Radiacion'] > 630
    return df_nasa

# 📊 Cargar y mostrar datos
df_historico = cargar_datos_locales()
df_nasa = cargar_datos_nasa()

# 📅 Unificar para comparar
df_historico['Periodo'] = '2008'
df_nasa['Periodo'] = '2024-2025'
df_comparado = pd.concat([
    df_historico[['Fecha', 'Radiacion', 'Humedad', 'Periodo']],
    df_nasa[['Fecha', 'Radiacion', 'Humedad', 'Periodo']]
]).reset_index(drop=True)

# 🔍 Mostrar dataframes
with st.expander("📄 Ver datos combinados"):
    st.dataframe(df_comparado)

# 📈 Gráfica radiación vs humedad
st.subheader("📈 Radiación vs Humedad Relativa")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_historico['Fecha'], df_historico['Radiacion'], label='Radiación 2008', color='orange')
ax.plot(df_historico['Fecha'], df_historico['Humedad'], label='Humedad 2008', color='blue')
ax.axhline(630, color='red', linestyle='--', label='Umbral 630 W/m²')
ax.set_title("Radiación vs Humedad - 2008")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# 🔥 Heatmap de correlación
st.subheader("🔗 Correlación entre variables (2008)")
fig2, ax2 = plt.subplots()
sns.heatmap(df_historico[['Radiacion', 'Presion', 'Humedad']].corr(), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

# 📊 Boxplot comparativo
st.subheader("📦 Comparativa de Radiación y Humedad")
fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_comparado, x='Periodo', y='Radiacion', palette='Oranges', ax=ax3)
ax3.axhline(630, color='red', linestyle='--', label='Umbral')
ax3.set_title('Radiación por Periodo')
st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df_comparado, x='Periodo', y='Humedad', palette='Blues', ax=ax4)
ax4.set_title('Humedad por Periodo')
st.pyplot(fig4)

# 📍 Mapa con promedio de radiación
st.subheader("🗺️ Mapa con Promedio Radiación 2024-2025")
promedio = df_nasa['Radiacion'].mean()
m = folium.Map(location=[-0.22, -78.36], zoom_start=9)
folium.Circle([ -0.22, -78.36], radius=30000, popup=f"Promedio: {promedio:.1f} W/m²", color='orange',
              fill=True, fill_opacity=0.4).add_to(m)
folium_static(m)

# 📤 Exportar CSV
csv = df_comparado.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Descargar CSV Comparado", data=csv, file_name="comparacion_mica.csv", mime='text/csv')
