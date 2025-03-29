from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import logging
import time

# Logging para generar logs y detectar el proceso tras consultar la API
logging.basicConfig(
    filename="aplicativo.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# IMPORTACION DEL DATASEOT Y ELECCION DE VARAIBLES
df = pd.read_csv("housePrice.csv")
#conversion de la columna area en un valor numerico
df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
#Eliminar valores nilos en Area y Address
df = df.dropna(subset=["Area", "Address"])

# VARIABLES
variables_secundarias = ["Area", "Room", "Parking", "Warehouse", "Elevator", "Address"]
variable_objetivo = "Price(USD)"
X = df[variables_secundarias]
y = df[variable_objetivo]

columnas_categoricas = ["Address"]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas)],
    remainder="passthrough"
)
# Aplicacion de pipeline para aplicar el preprosesamiento de datos, definir el modelo y definir el valor del hiperparametro
pipeline = make_pipeline(
    preprocessor,
    RandomForestRegressor(n_estimators=100, random_state=42)
)
#Entrenamiento de modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

#IMPLEMENTACION DE FASTAPI
app = FastAPI(title="API de Predicción de Precio de Vivienda con Logging")

class datosCasa(BaseModel):
    Area: float
    Room: int
    Parking: bool
    Warehouse: bool
    Elevator: bool
    Address: str

#Middleware que registra cada peticion que se le hace a la API por medio del puerto HTTP
@app.middleware("http")

async def log_requests(request: Request, call_next):

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    # Modifica el archivo log tras la peticion
    logging.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.4f}s")
    return response


#ENDPOINTS
# endpoint raiz (se activa sipre que nos comuniquemos con la ruta web de la API)
@app.get("/")
def raiz():
    logging.info("Acceso al endpoint de la API")
    return {"mensaje": "API para predecir el precio de una casa en USD - by Juan Jose Rincon"}

#endpoint para visualizar el dataset
@app.get("/verDataset")
def ver_dataset():
    logging.info("ver dataset completo")
    return df.to_dict(orient="records")

# endpoint de prediccion al recibir datos como:

#"Area": 145,
# "Room": 3,
# "Parking": true,
# "Warehouse": true,
# "Elevator": true,
# "Address": "Punak"

# se hace uso del modelo previamente entrenado y se predice el resultado 

@app.post("/predecir")
def predecir(datosEvaluar: datosCasa):
    try:
        #se convietne los datos a evaluar en un dataframe
        df_entrada= pd.DataFrame([datosEvaluar.dict()])
        logging.info(f"\nSolicitud recibida - DATOS recibidos: {datosEvaluar.dict()}")
        #Se le envian los datos al pipeline para que ponga aprueba el modleo entredao y realice la prediccion
        prediction = pipeline.predict(df_entrada)[0]
        logging.info(f"Predicción generada por parte de modelo: {prediction}")
        #Respuesta por parte de la api tras la consulta
        return {"Precio estimado de la casa USD: ": round(prediction, 2)}
    except Exception as e:
        logging.error(f"Error en predicción: {e}", exc_info=True)
        return {"error": "Error al realizar la "}
