# nasa_api/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
from datetime import datetime
from fastapi.responses import JSONResponse

app = FastAPI()

# Permitir acceso desde cualquier origen (útil para Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/nasa-datos")
def obtener_datos_nasa(
    lat: float = Query(-0.35),
    lon: float = Query(-78.35),
    start_date: str = Query("20240101"),
    end_date: str = Query(datetime.today().strftime("%Y%m%d"))
):
    try:
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point"
            f"?parameters=ALLSKY_SFC_SW_DWN,RH2M"
            f"&community=AG&longitude={lon}&latitude={lat}"
            f"&start={start_date}&end={end_date}&format=JSON"
        )
        response = requests.get(url)
        data = response.json()

        df = pd.DataFrame({
            'Fecha': list(data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].keys()),
            'Radiacion': list(data['properties']['parameter']['ALLSKY_SFC_SW_DWN'].values()),
            'Humedad': list(data['properties']['parameter']['RH2M'].values())
        })
        df['Fecha'] = pd.to_datetime(df['Fecha'])

        return JSONResponse(df.to_dict(orient="records"))
    except Exception as e:
        return {"error": str(e)}
