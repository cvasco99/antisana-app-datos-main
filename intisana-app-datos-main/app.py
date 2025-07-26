import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import folium
from streamlit_folium import st_folium, folium_static
from datetime import datetime
import google.generativeai as genai
import time
import pyttsx3
from PIL import Image
import sqlite3
import matplotlib.dates as mdates
import numpy as np




# 🔧 Configuración inicial
st.set_page_config(page_title="☀️ Comparador Solar Mica", layout="wide")
st.title("☀️ Comparador Radiación Solar y Humedad - Mica, Ecuador")

# 🔐 API Key desde secrets.toml
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
modelo_gemini = genai.GenerativeModel("gemini-2.0-flash")

# 🧭 Tabs principales
tabs = st.tabs([
    "📤 Comparar con archivo Excel", 
    "📈 Comparar histórico vs NASA", 
    "🤖 Asistente IA "
])

# Función para lectura por voz
def hablar(texto):
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(texto)
    engine.runAndWait()

# Guardar historial en SQLite
def guardar_mensaje_en_bd(role, content):
    conn = sqlite3.connect('chat_gemini.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS mensajes
                 (rol TEXT, contenido TEXT, fecha DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute("INSERT INTO mensajes (rol, contenido) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

# 📤 TAB 1: Cargar Excel personalizado
with tabs[0]:
    st.subheader("📤 Subir datos históricos personalizados")

    archivo = st.file_uploader("Sube archivo .xlsx con datos históricos (radiación y/o humedad)", type=["xlsx"])
    if archivo:
        try:
            df_hist = pd.read_excel(archivo, header=11)
            df_hist.columns = df_hist.columns.str.strip()
            df_hist['Fecha'] = pd.to_datetime(df_hist['Fecha'], errors='coerce')
            df_hist.dropna(subset=['Fecha'], inplace=True)

            # Detectar columnas radiacion y humedad
            tiene_radiacion = 'Radiacion' in df_hist.columns or 'Valor' in df_hist.columns
            tiene_humedad = 'Humedad' in df_hist.columns

            # Preparar columnas según lo que hay
            if tiene_radiacion and 'Valor' in df_hist.columns:
                df_hist = df_hist.rename(columns={'Valor': 'Radiacion'})
            if not tiene_humedad and 'Humedad' not in df_hist.columns:
                df_hist['Humedad'] = None
            if 'Radiacion' not in df_hist.columns:
                df_hist['Radiacion'] = None

            df_hist['Periodo'] = 'Histórico'


            # Filtrar rango de fechas según el archivo
            fecha_min, fecha_max = df_hist['Fecha'].min(), df_hist['Fecha'].max()
            rango = st.date_input("🗓️ Filtrar fechas", [fecha_min, fecha_max], min_value=fecha_min, max_value=fecha_max)
            rango_min = pd.to_datetime(rango[0])
            rango_max = pd.to_datetime(rango[1])
            df_hist = df_hist[(df_hist['Fecha'] >= rango_min) & (df_hist['Fecha'] <= rango_max)]

            st.info("🔄 Obteniendo datos actuales NASA POWER...")
            lat, lon = -0.22, -78.36
            # Ajustar rango NASA a filtro elegido para mejor coherencia
            start_nasa = rango_min.strftime('%Y%m%d')
            end_nasa = rango_max.strftime('%Y%m%d')
            url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
                   f"parameters=ALLSKY_SFC_SW_DWN,RH2M&community=AG&longitude={lon}&latitude={lat}"
                   f"&start={start_nasa}&end={end_nasa}&format=JSON")
            nasa = requests.get(url).json()

            df_nasa = pd.DataFrame({
                'Fecha': pd.to_datetime(list(nasa['properties']['parameter']['ALLSKY_SFC_SW_DWN'].keys())),
                'Radiacion': [val * 10 for val in nasa['properties']['parameter']['ALLSKY_SFC_SW_DWN'].values()],
                'Humedad': list(nasa['properties']['parameter']['RH2M'].values())
            })

            df_nasa['Periodo'] = 'NASA 2024–2025'


            # Completar columnas según datos presentes
            if not tiene_radiacion:
                df_hist['Radiacion'] = np.nan
                df_nasa['Radiacion'] = np.nan
            if not tiene_humedad:
                df_hist['Humedad'] = np.nan
                df_nasa['Humedad'] = np.nan

            # Concatenar datos
            df = pd.concat([df_hist, df_nasa], ignore_index=True).reset_index(drop=True)

            # Filtrar rango fecha general para mantener coherencia en gráficas
            df = df[(df['Fecha'] >= rango_min) & (df['Fecha'] <= rango_max)]

            # ---- GRAFICOS RADIACION ----
            if tiene_radiacion:
                st.subheader("📊 Series Temporales de Radiación")
                fig, ax = plt.subplots(figsize=(14, 5))
                sns.lineplot(data=df.dropna(subset=['Radiacion']), x='Fecha', y='Radiacion', hue='Periodo', ax=ax)

                # Ajustar rango Y dinámico
                radi_min = df['Radiacion'].min()
                radi_max = df['Radiacion'].max()
                ymin = max(0, radi_min - 20)
                ymax = radi_max + 20
                ax.set_ylim(ymin, ymax)

                # Ajustar rango X al filtro y mostrar meses con formato
                ax.set_xlim(rango_min, rango_max)
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=45)

                ax.set_ylabel("Radiación (W/m²)")
                ax.set_xlabel("Fecha")
                ax.axhline(630, color='red', linestyle='--', label='Umbral 630 W/m²')
                ax.legend()
                ax.grid()
                st.pyplot(fig)

                st.subheader("📦 Boxplot de Radiación")
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df.dropna(subset=['Radiacion']), x='Periodo', y='Radiacion', palette='Oranges', ax=ax2)
                radi_min_box = df['Radiacion'].min()
                radi_max_box = df['Radiacion'].max()
                margin = (radi_max_box - radi_min_box) * 0.1
                ax2.set_ylim(radi_min_box - margin, radi_max_box + margin)
                ax2.axhline(630, color='red', linestyle='--', label='Umbral 630 W/m²')
                ax2.legend()
                st.pyplot(fig2)

                st.subheader("🔴 % Días con Radiación Alta (Top días sin filtro específico)")
                # Calcular porcentaje días con radiación mayor o igual al promedio top 10 por periodo
                porcentajes = {}
                for periodo in df['Periodo'].unique():
                    df_p = df[df['Periodo'] == periodo].dropna(subset=['Radiacion'])
                    if df_p.empty:
                        porcentajes[periodo] = 0
                        continue
                    top_10_promedio = df_p.nlargest(10, 'Radiacion')['Radiacion'].mean()
                    dias_altos = df_p[df_p['Radiacion'] >= top_10_promedio]
                    porcentaje = len(dias_altos) / len(df_p) * 100
                    porcentajes[periodo] = porcentaje
                st.bar_chart(pd.Series(porcentajes))

                st.subheader("🔥 Top 10 Días con Mayor Radiación")
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                colores = {'Histórico': 'orange', 'NASA 2024–2025': 'red'}
                for periodo in ['Histórico', 'NASA 2024–2025']:
                    df_p = df[df['Periodo'] == periodo].dropna(subset=['Radiacion'])
                    top10 = df_p.nlargest(10, 'Radiacion')
                    labels = [f"{periodo} - Día {i+1}" for i in range(len(top10))]
                    ax3.bar(labels, top10['Radiacion'], color=colores.get(periodo, 'grey'))
                ax3.set_ylabel("Radiación (W/m²)")
                ax3.set_title("Top 10 Días con Mayor Radiación")
                plt.xticks(rotation=45, ha='right')
                ax3.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig3)

            # ---- GRAFICOS HUMEDAD ----
            if tiene_humedad:
                st.subheader("📊 Series Temporales de Humedad")
                fig4, ax4 = plt.subplots(figsize=(14, 5))
                sns.lineplot(data=df.dropna(subset=['Humedad']), x='Fecha', y='Humedad', hue='Periodo', ax=ax4)
                ax4.set_xlim(rango_min, rango_max)
                ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=45)
                ax4.set_ylabel("Humedad (%)")
                ax4.set_xlabel("Fecha")
                ax4.grid()
                st.pyplot(fig4)

                st.subheader("📦 Boxplot de Humedad")
                fig5, ax5 = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=df.dropna(subset=['Humedad']), x='Periodo', y='Humedad', palette='Blues', ax=ax5)
                st.pyplot(fig5)

            # ---- GRAFICO COMBINADO SI HAY AMBOS ----
            if tiene_radiacion and tiene_humedad:
                st.subheader("📊 Radiación y Humedad combinados")
                fig6, ax6 = plt.subplots(figsize=(14, 5))
                ax6_2 = ax6.twinx()

                df_r = df.dropna(subset=['Radiacion'])
                df_h = df.dropna(subset=['Humedad'])

                sns.lineplot(data=df_r, x='Fecha', y='Radiacion', hue='Periodo', ax=ax6, legend=False,
                             palette=['orange', 'red'])
                sns.lineplot(data=df_h, x='Fecha', y='Humedad', hue='Periodo', ax=ax6_2, legend=False,
                             palette=['blue', 'cyan'])

                ax6.set_xlim(rango_min, rango_max)
                ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                plt.xticks(rotation=45)

                ax6.set_ylabel("Radiación (W/m²)")
                ax6_2.set_ylabel("Humedad (%)")
                ax6.set_xlabel("Fecha")
                ax6.grid()
                st.pyplot(fig6)

            # Descargar CSV
            st.subheader("📥 Descargar comparación")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Descargar CSV", csv, f"comparacion_{datetime.now().date()}.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error al procesar el archivo: {e}")


# 📈 TAB 2: Comparar 2008 vs NASA
# 📈 TAB 2: Comparar 2008 vs NASA vs Actual (Separado)
with tabs[1]:
    st.subheader("📈 Comparativas por periodo, fuente y variable")

    import matplotlib.dates as mdates
    import folium
    from streamlit_folium import folium_static

    # --- Funciones de carga ---
    @st.cache_data
    def cargar_datos():
        df_c09 = pd.read_excel("C09-Mica_Campamento_Radiación_solar-Diario.xlsx", header=11)[['Fecha', 'Valor']]
        df_c10 = pd.read_excel("C10-La_Mica_Presa_Radiación_solar-Diario.xlsx", header=11)[['Fecha', 'Valor']]
        df_c09.columns = df_c10.columns = ['Fecha', 'Radiacion']
        df_c09['Fecha'] = pd.to_datetime(df_c09['Fecha'], errors='coerce')
        df_c10['Fecha'] = pd.to_datetime(df_c10['Fecha'], errors='coerce')
        df_c09['Periodo'] = 'C09 2008'
        df_c10['Periodo'] = 'C10 2024'

        df_h09 = pd.read_excel("C09-Mica_Campamento_Humedad_relativa-Diario.xlsx", header=11)[['Fecha', 'Valor']]
        df_h10 = pd.read_excel("C10-La_Mica_Presa_Humedad_relativa-Diario.xlsx", header=11)[['Fecha', 'Valor']]
        df_h09.columns = df_h10.columns = ['Fecha', 'Humedad']
        df_h09['Fecha'] = pd.to_datetime(df_h09['Fecha'], errors='coerce')
        df_h10['Fecha'] = pd.to_datetime(df_h10['Fecha'], errors='coerce')
        df_h09['Periodo'] = 'C09 2008'
        df_h10['Periodo'] = 'C10 2024'

        return df_c09, df_c10, df_h09, df_h10

    @st.cache_data
    def cargar_nasa():
        url = ("https://power.larc.nasa.gov/api/temporal/daily/point?"
               "parameters=ALLSKY_SFC_SW_DWN,RH2M&community=AG&longitude=-78.21&latitude=-0.55&"
               "start=20080101&end=20250715&format=JSON")
        data = requests.get(url).json()['properties']['parameter']
        df_nasa = pd.DataFrame({
            'Fecha': pd.to_datetime(list(data['ALLSKY_SFC_SW_DWN'].keys())),
            'Radiacion': [val * 10 for val in data['ALLSKY_SFC_SW_DWN'].values()],
            'Humedad': list(data['RH2M'].values())
        })
        df_nasa['Periodo'] = df_nasa['Fecha'].dt.year.map(lambda x: 'NASA 2008' if x == 2008 else ('NASA 2024' if x == 2024 else None))
        return df_nasa.dropna(subset=['Periodo'])

    df_c09, df_c10, df_h09, df_h10 = cargar_datos()
    df_nasa = cargar_nasa()

    # 1. Series temporales de Radiación (filtradas por año)
    st.markdown("### 📈 1. Series temporales de Radiación")
    for anio, (local, nasa) in {'2008': ('C09 2008', 'NASA 2008'), '2024': ('C10 2024', 'NASA 2024')}.items():
        df_local = (df_c09 if anio == '2008' else df_c10)
        df_local = df_local[df_local['Fecha'].dt.year == int(anio)]
        df_nasa_filtrado = df_nasa[(df_nasa['Periodo'] == nasa) & (df_nasa['Fecha'].dt.year == int(anio))]
        df_plot = pd.concat([df_local, df_nasa_filtrado], ignore_index=True)

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.lineplot(data=df_plot, x='Fecha', y='Radiacion', hue='Periodo', ax=ax)
        ax.axhline(630, color='red', linestyle='--', label='Umbral 630')
        ax.set_title(f'Radiación Solar Diario - {anio}')
        ax.grid()
        ax.set_ylabel("Radiación (W/m²)")
        ax.set_xlabel("Fecha")
        st.pyplot(fig)

    # 2. Series temporales de Humedad (filtradas por año)
    st.markdown("### 💧 2. Series temporales de Humedad")
    for anio, (local_df, nasa_tag) in {'2008': (df_h09, 'NASA 2008'), '2024': (df_h10, 'NASA 2024')}.items():
        df_local = local_df.copy()
        df_local = df_local[df_local['Fecha'].dt.year == int(anio)]
        df_nasa_h = df_nasa[(df_nasa['Periodo'] == nasa_tag) & (df_nasa['Fecha'].dt.year == int(anio))][['Fecha', 'Humedad']].copy()
        df_local['Periodo'] = df_local['Periodo'].iloc[0]
        df_nasa_h['Periodo'] = nasa_tag
        df_plot = pd.concat([df_local, df_nasa_h], ignore_index=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.lineplot(data=df_plot, x='Fecha', y='Humedad', hue='Periodo', ax=ax)
        ax.set_title(f'Humedad Relativa Diario - {anio}')
        ax.grid()
        ax.set_ylabel("Humedad (%)")
        ax.set_xlabel("Fecha")
        st.pyplot(fig)

    # 3. Promedios Mensuales de Radiación con eje X mejorado (sin filtro anual)
    st.markdown("### 📅 3. Promedios mensuales de Radiación")
    for anio in ['2008', '2024']:
        df_local = df_c09 if anio == '2008' else df_c10
        tag_nasa = 'NASA 2008' if anio == '2008' else 'NASA 2024'

        df_local['AñoMes'] = df_local['Fecha'].dt.to_period('M').astype(str)
        df_n = df_nasa[df_nasa['Periodo'] == tag_nasa]
        df_n['AñoMes'] = df_n['Fecha'].dt.to_period('M').astype(str)

        df_plot = pd.concat([
            df_local.groupby('AñoMes')['Radiacion'].mean().reset_index().assign(Periodo=df_local['Periodo'].iloc[0]),
            df_n.groupby('AñoMes')['Radiacion'].mean().reset_index().assign(Periodo=tag_nasa)
        ], ignore_index=True).reset_index(drop=True)

        df_plot['AñoMes_dt'] = pd.to_datetime(df_plot['AñoMes'])
        df_plot = df_plot.sort_values('AñoMes_dt')

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.lineplot(data=df_plot, x='AñoMes_dt', y='Radiacion', hue='Periodo', marker='o', ax=ax)
        ymin = df_plot['Radiacion'].min() - 20
        ymax = df_plot['Radiacion'].max() + 20
        ax.set_ylim(ymin, ymax)
        ax.axhline(630, color='red', linestyle='--')
        ax.set_title(f'Promedios Mensuales de Radiación - {anio}')
        ax.set_ylabel("Radiación (W/m²)")
        ax.set_xlabel("Mes")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        ax.grid()
        st.pyplot(fig)

    # 4. Boxplots de Radiación con filtro anual
    st.markdown("### 📦 4. Boxplots de Radiación")
    for anio in ['2008', '2024']:
        df_r = df_c09 if anio == '2008' else df_c10
        df_r = df_r[df_r['Fecha'].dt.year == int(anio)].copy()
        tag_nasa = 'NASA 2008' if anio == '2008' else 'NASA 2024'
        df_n = df_nasa[(df_nasa['Periodo'] == tag_nasa) & (df_nasa['Fecha'].dt.year == int(anio))].copy()
        df_r['Periodo'] = df_r['Periodo'].iloc[0]
        df_box = pd.concat([df_r, df_n], ignore_index=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_box, x='Periodo', y='Radiacion', palette='Oranges', ax=ax)
        ax.axhline(630, color='red', linestyle='--')
        ax.set_title(f'Boxplot de Radiación Solar - {anio}')
        ax.set_ylabel("Radiación (W/m²)")
        st.pyplot(fig)

    # 5. Boxplots de Humedad (sin cambio)
    st.markdown("### 📦 5. Boxplots de Humedad")
    for anio in ['2008', '2024']:
        df_h = df_h09 if anio == '2008' else df_h10
        tag_nasa = 'NASA 2008' if anio == '2008' else 'NASA 2024'
        df_h = df_h.copy()
        df_n = df_nasa[df_nasa['Periodo'] == tag_nasa][['Fecha', 'Humedad']].copy()
        df_h['Periodo'] = df_h['Periodo'].iloc[0]
        df_n['Periodo'] = tag_nasa
        df_box = pd.concat([df_h, df_n], ignore_index=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df_box, x='Periodo', y='Humedad', palette='Blues', ax=ax)
        ax.set_title(f'Boxplot de Humedad Relativa - {anio}')
        ax.set_ylabel("Humedad (%)")
        st.pyplot(fig)

    # 6. Días con Radiación Más Alta (Top 10)
    st.markdown("### 🔥 6. Días con Radiación Más Alta (Top 10)")
    for anio in ['2008', '2024']:
        df_r = df_c09 if anio == '2008' else df_c10
        df_r = df_r[df_r['Fecha'].dt.year == int(anio)].copy()
        tag_nasa = 'NASA 2008' if anio == '2008' else 'NASA 2024'
        df_n = df_nasa[(df_nasa['Periodo'] == tag_nasa) & (df_nasa['Fecha'].dt.year == int(anio))].copy()

        df_r['Periodo'] = df_r['Periodo'].iloc[0]
        df_n['Periodo'] = tag_nasa

        top_local = df_r.nlargest(10, 'Radiacion')
        top_nasa = df_n.nlargest(10, 'Radiacion')

        conteo = pd.Series({
            df_r['Periodo'].iloc[0]: len(top_local),
            tag_nasa: len(top_nasa)
        })

        promedio_top = pd.Series({
            df_r['Periodo'].iloc[0]: top_local['Radiacion'].mean(),
            tag_nasa: top_nasa['Radiacion'].mean()
        })

        fig, ax = plt.subplots(figsize=(8, 5))
        barras = ax.bar(conteo.index, conteo.values, color=['orange', 'red'])

        ax.set_title(f'Días con Radiación Más Alta (Top 10) - {anio}')
        ax.set_ylabel('Número de días')
        ax.set_ylim(0, 15)

        for bar, prom in zip(barras, promedio_top):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{int(height)} días\nPromedio: {prom:.1f} W/m²',
                    ha='center', va='bottom', fontsize=10)

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig)

    # 7. Mapa interactivo con boxplot y círculo promedio NASA
    st.markdown("### 🗺️ 7. Mapa con Promedios de Radiación y Boxplot de Humedad")

    # Boxplot de Humedad general combinado
    df_comparado = pd.concat([df_h09, df_h10, df_nasa[['Fecha', 'Humedad', 'Periodo']]], ignore_index=True)
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_comparado, x='Periodo', y='Humedad', palette='Blues', ax=ax4)
    ax4.set_title("Boxplot de Humedad Relativa - General")
    st.pyplot(fig4)

    # Mapa con círculo promedio NASA
    promedio_nasa = df_nasa['Radiacion'].mean()
    m = folium.Map(location=[-0.5406111, -78.2084602], zoom_start=9)
    folium.Circle(
        location=[-0.5406111, -78.2084602],
        radius=30000,
        popup=f"Promedio NASA: {promedio_nasa:.1f} W/m²",
        color='orange',
        fill=True,
        fill_opacity=0.4
    ).add_to(m)
    folium_static(m)

    # 8. Descargar CSV final
    st.markdown("### 💾 8. Descargar datos comparados")
    df_comparado_general = pd.concat([df_c09, df_c10, df_nasa], ignore_index=True).reset_index(drop=True)
    st.download_button(
        label="⬇️ Descargar CSV",
        data=df_comparado_general.to_csv(index=False).encode('utf-8'),
        file_name="datos_comparativos_radiacion.csv",
        mime='text/csv'
    )


# 🤖 TAB 3: Asistente con Gemini (IA de Google)
# 🤖 TAB 3: Asistente con Gemini (IA de Google)
with tabs[2]:
    import time
    import re

    st.markdown("## 🧠 Pregunta sobre Radiación Solar y Humedad")

    if "mensajes_gemini" not in st.session_state:
        st.session_state.mensajes_gemini = []
    if "esperando_respuesta" not in st.session_state:
        st.session_state.esperando_respuesta = False

    def limpiar_html(texto):
        return re.sub(r'</?div[^>]*>', '', texto)

    chat_container = st.container()

    with chat_container:
        st.markdown('<div id="chat-history" style="max-height: 550px; overflow-y: auto; padding: 10px;">', unsafe_allow_html=True)
        for i, mensaje in enumerate(st.session_state.mensajes_gemini):
            contenido_limpio = limpiar_html(mensaje['content'])
            if mensaje["role"] == "user":
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end;">
                    <div class="chat-bubble chat-user"><strong>{contenido_limpio}</strong></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                id_texto = f"respuesta-{i}"
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; flex-direction: column;">
                    <div class="chat-bubble chat-assistant" id="{id_texto}">{contenido_limpio}</div>
                    <div style="text-align: right; margin-top: 4px; margin-bottom: 10px;">
                        <button onclick="copiarTexto('{id_texto}')" style="
                            font-size: 0.85rem;
                            padding: 4px 10px;
                            border: none;
                            background-color: transparent;
                            color: var(--btn-text, #0d6efd);
                            cursor: pointer;
                        ">📋 Copiar</button>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    pregunta = st.chat_input("Escribe tu pregunta sobre radiación solar...")

    acciones_placeholder = st.empty()
    respuesta_placeholder = st.empty()

    if pregunta and not st.session_state.esperando_respuesta:
        with chat_container:
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end;">
                <div class="chat-bubble chat-user"><strong>{pregunta}</strong></div>
            </div>
            """, unsafe_allow_html=True)

        st.session_state.mensajes_gemini.append({"role": "user", "content": pregunta})
        st.session_state.esperando_respuesta = True

        try:
            parts = []
            for msg in st.session_state.mensajes_gemini:
                prefijo = "Usuario:" if msg["role"] == "user" else "IA:"
                parts.append({"text": f"{prefijo} {msg['content']}"})
            contents = [{"parts": parts}]

            respuesta = modelo_gemini.generate_content(contents=contents)
            texto_respuesta = respuesta.text if hasattr(respuesta, "text") else respuesta.candidates[0].content.parts[0].text
            texto_respuesta = limpiar_html(texto_respuesta)

            texto_progresivo = ""
            placeholder = respuesta_placeholder.empty()
            for char in texto_respuesta:
                texto_progresivo += char
                placeholder.markdown(f"""
                <div style="display: flex; justify-content: flex-start;">
                    <div class="chat-bubble chat-assistant">{texto_progresivo}▌</div>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.01)

            placeholder.markdown(f"""
            <div style="display: flex; justify-content: flex-start; flex-direction: column;">
                <div class="chat-bubble chat-assistant">{texto_progresivo}</div>
                <div style="text-align: right; margin-top: 4px; margin-bottom: 10px;">
                    <button onclick="copiarTexto('respuesta-final')" style="
                        font-size: 0.85rem;
                        padding: 4px 10px;
                        border: none;
                        background-color: transparent;
                        color: var(--btn-text, #0d6efd);
                        cursor: pointer;
                    ">📋 Copiar</button>
                </div>
            </div>
            <div id="respuesta-final" style="display: none;">{texto_progresivo}</div>
            """, unsafe_allow_html=True)

            st.session_state.mensajes_gemini.append({"role": "assistant", "content": texto_respuesta})
            st.session_state.esperando_respuesta = False

            with acciones_placeholder.container():
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("")  # Puedes dejar espacio aquí si quieres otros botones
                with col2:
                    if st.button("➕ Nuevo chat"):
                        st.session_state.mensajes_gemini = []
                        st.session_state.esperando_respuesta = False
                        st.rerun()

        except Exception as e:
            st.error(f"❌ Error al consultar a Gemini: {e}")
            st.session_state.esperando_respuesta = False



    # 💄 ESTILOS
    st.markdown("""
    <style>
    .block-container { padding-bottom: 100px !important; }

    div[data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 16px;
        left: 50%;
        transform: translateX(-50%);
        width: 640px;
        max-width: 95vw;
        border-radius: 24px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        padding: 8px 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        z-index: 9999;
        background-color: var(--input-bg);
        border: 1px solid var(--input-border);
    }

    textarea[data-testid="stChatInputTextArea"] {
        flex-grow: 1;
        border: none;
        padding: 10px 16px !important;
        border-radius: 18px !important;
        font-size: 1rem !important;
        background-color: var(--textarea-bg);
        color: var(--textarea-text);
        resize: none !important;
        outline: none !important;
    }

    button[data-testid="stChatInputSubmitButton"] {
        background-color: var(--btn-bg);
        color: var(--btn-text);
        border: none;
        font-size: 1.3rem;
        padding: 6px 10px;
        border-radius: 18px;
        cursor: pointer;
    }

    html[data-theme="light"] {
        --input-bg: rgba(245, 245, 245, 0.95);
        --input-border: rgba(0, 0, 0, 0.1);
        --textarea-bg: #ffffff;
        --textarea-text: #000000;
        --btn-bg: #0d6efd;
        --btn-text: #ffffff;
        --user-bubble-bg: #0d6efd;
        --user-bubble-text: #ffffff;
        --assistant-bubble-bg: #e5e5ea;
        --assistant-bubble-text: #000000;
    }

    html[data-theme="dark"] {
        --input-bg: rgba(32, 33, 36, 0.95);
        --input-border: rgba(255, 255, 255, 0.1);
        --textarea-bg: #303134;
        --textarea-text: #e8eaed;
        --btn-bg: #0d6efd;
        --btn-text: #ffffff;
        --user-bubble-bg: #0d6efd;
        --user-bubble-text: #ffffff;
        --assistant-bubble-bg: #3a3b3c;
        --assistant-bubble-text: #e8eaed;
    }

    .chat-bubble {
        max-width: 70%;
        padding: 12px 16px;
        font-size: 1.05rem;
        border-radius: 18px;
        white-space: pre-wrap;
        margin-bottom: 6px;
        line-height: 1.25;
    }

    .chat-user {
        background-color: var(--user-bubble-bg);
        color: var(--user-bubble-text);
        border-radius: 18px 18px 0 18px;
        align-self: flex-end;
    }

    .chat-assistant {
        background-color: var(--assistant-bubble-bg);
        color: var(--assistant-bubble-text);
        border-radius: 18px 18px 18px 0;
        align-self: flex-start;
    }

    .chat-attach {
        width: 24px;
        height: 24px;
        cursor: pointer;
        position: relative;
    }

    .chat-attach input[type="file"] {
        position: absolute;
        top: 0;
        left: 0;
        opacity: 0;
        width: 24px;
        height: 24px;
        cursor: pointer;
    }

    .chat-attach img {
        width: 24px;
        height: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Copiar al portapapeles con JS
    st.markdown("""
    <script>
    function copiarTexto(id) {
        const contenido = document.getElementById(id).innerText;
        navigator.clipboard.writeText(contenido).then(() => {
            alert("✅ Respuesta copiada al portapapeles");
        }).catch(err => {
            alert("❌ Error al copiar");
        });
    }
    </script>
    """, unsafe_allow_html=True)

    # Scroll al fondo
    st.markdown("""
    <script>
    const chatHistory = document.getElementById('chat-history');
    if(chatHistory){
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
    </script>
    """, unsafe_allow_html=True)
