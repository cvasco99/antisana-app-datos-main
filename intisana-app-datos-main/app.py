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


# 🔧 Configuración inicial
st.set_page_config(page_title="☀️ Comparador Solar Mica", layout="wide")
st.title("☀️ Comparador Radiación Solar y Humedad - Mica, Ecuador")

# 🔐 API Key desde secrets.toml
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
modelo_gemini = genai.GenerativeModel("gemini-2.0-flash")

# 🧭 Tabs principales
tabs = st.tabs([
    "📤 Comparar con archivo Excel", 
    "📈 Comparar histórico 2008 vs NASA", 
    "🤖 Asistente IA (Gemini)"
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

    archivo = st.file_uploader("Sube archivo .xlsx con datos históricos (radiación y humedad)", type=["xlsx"])
    if archivo:
        try:
            df_hist = pd.read_excel(archivo, header=11)
            df_hist.columns = df_hist.columns.str.strip()
            df_hist = df_hist[['Fecha', 'Valor']].rename(columns={'Valor': 'Radiacion'})
            df_hist['Fecha'] = pd.to_datetime(df_hist['Fecha'])
            df_hist['Periodo'] = 'Histórico'
            df_hist['Humedad'] = None

            fecha_min, fecha_max = df_hist['Fecha'].min(), df_hist['Fecha'].max()
            rango = st.date_input("🗓️ Filtrar fechas", [fecha_min, fecha_max],
                                  min_value=fecha_min, max_value=fecha_max)
            df_hist = df_hist[(df_hist['Fecha'] >= pd.to_datetime(rango[0])) & (df_hist['Fecha'] <= pd.to_datetime(rango[1]))]

            st.info("🔄 Obteniendo datos actuales NASA POWER...")
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

            df = pd.concat([df_hist, df_nasa]).reset_index(drop=True)
            df['Radiacion_Alta'] = df['Radiacion'] > 630

            st.subheader("📊 Boxplot Radiación")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x='Periodo', y='Radiacion', palette='Oranges', ax=ax)
            plt.axhline(630, color='red', linestyle='--', label='Umbral 630 W/m²')
            plt.legend()
            st.pyplot(fig)

            st.subheader("🔴 % Días con radiación > 630 W/m²")
            porcentajes = df.groupby('Periodo')['Radiacion_Alta'].mean() * 100
            st.bar_chart(porcentajes)

            st.subheader("📥 Descargar comparación")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Descargar CSV", csv, f"comparacion_{datetime.now().date()}.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error al procesar el archivo: {e}")

# 📈 TAB 2: Comparar 2008 vs NASA
with tabs[1]:
    st.subheader("📈 Comparativa histórica (2008) vs NASA actual")

    @st.cache_data
    def cargar_datos_locales():
        df_r = pd.read_excel("C09-Mica_Campamento_Radiación_solar-Diario.xlsx", header=11)
        df_p = pd.read_excel("C09-Mica_Campamento_Presion_atmosférica-Diario.xlsx", header=11)
        df_h = pd.read_excel("C09-Mica_Campamento_Humedad_relativa-Diario.xlsx", header=11)
        df_r = df_r[['Fecha', 'Valor']].rename(columns={'Valor': 'Radiacion'})
        df_p = df_p[['Fecha', 'Valor']].rename(columns={'Valor': 'Presion'})
        df_h = df_h[['Fecha', 'Valor']].rename(columns={'Valor': 'Humedad'})
        for df in [df_r, df_p, df_h]:
            df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
            df.dropna(subset=['Fecha'], inplace=True)
        df_total = df_r.merge(df_p, on='Fecha').merge(df_h, on='Fecha')
        df_total['Radiacion_Alta'] = df_total['Radiacion'] > 630
        return df_total

    @st.cache_data
    def cargar_datos_nasa():
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,RH2M&community=AG&longitude=-78.36&latitude=-0.22&start=20240101&end=20250622&format=JSON"
        data = requests.get(url).json()['properties']['parameter']
        df = pd.DataFrame({
            'Fecha': pd.to_datetime(list(data['ALLSKY_SFC_SW_DWN'].keys())),
            'Radiacion': list(data['ALLSKY_SFC_SW_DWN'].values()),
            'Humedad': list(data['RH2M'].values())
        })
        df['Radiacion_Alta'] = df['Radiacion'] > 630
        return df

    df_2008 = cargar_datos_locales()
    df_nasa = cargar_datos_nasa()

    df_2008['Periodo'] = '2008'
    df_nasa['Periodo'] = '2024–2025'
    df_comparado = pd.concat([
        df_2008[['Fecha', 'Radiacion', 'Humedad', 'Presion', 'Periodo']],
        df_nasa[['Fecha', 'Radiacion', 'Humedad', 'Periodo']]
    ], ignore_index=True)
    st.dataframe(df_comparado)

    st.markdown("### 📊 Radiación y humedad - Año 2008")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_2008['Fecha'], df_2008['Radiacion'], label='Radiación', color='orange')
    ax.plot(df_2008['Fecha'], df_2008['Humedad'], label='Humedad', color='blue')
    ax.axhline(630, color='red', linestyle='--')
    ax.legend(); ax.grid(); st.pyplot(fig)

    st.markdown("### 🔥 Correlación entre variables")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df_2008[['Radiacion', 'Presion', 'Humedad']].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.markdown("### 📦 Boxplots comparativos")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_comparado, x='Periodo', y='Radiacion', palette='Oranges', ax=ax3)
    ax3.axhline(630, color='red', linestyle='--')
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_comparado, x='Periodo', y='Humedad', palette='Blues', ax=ax4)
    st.pyplot(fig4)

    promedio = df_nasa['Radiacion'].mean()
    m = folium.Map(location=[-0.22, -78.36], zoom_start=9)
    folium.Circle([-0.22, -78.36], radius=30000, popup=f"Promedio: {promedio:.1f} W/m²",
                  color='orange', fill=True, fill_opacity=0.4).add_to(m)
    folium_static(m)

    st.download_button("⬇️ Descargar CSV Comparado", data=df_comparado.to_csv(index=False).encode('utf-8'),
                       file_name="comparacion_mica.csv", mime='text/csv')


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

    # 📎 Ícono visual de adjuntar (sin funcionalidad todavía)
    st.markdown("""
    <div style="position: fixed; bottom: 22px; left: calc(50% - 330px); z-index: 9999;">
        <label class="chat-attach">
            <img src="https://cdn-icons-png.flaticon.com/512/1828/1828817.png" alt="Adjuntar archivo">
            <input type="file" />
        </label>
    </div>
    """, unsafe_allow_html=True)

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
