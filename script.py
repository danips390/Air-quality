import os
import json
import pandas as pd
import numpy as np
import requests
import geojson
import shapefile
from shapely.geometry import shape, Point
from scipy.spatial import Delaunay
from datetime import datetime

# Parámetros
API_KEY = os.getenv("API_KEY_PURPLEAIR")
CSV_FILE = 'sensores_detectados.csv'
SALIDA_GEOJSON_SENSORES = 'sensores.geojson'
SALIDA_GEOJSON_COLONIAS_PM25 = 'AQ_PM25.geojson'
SALIDA_GEOJSON_COLONIAS_PM10 = 'AQ_PM10.geojson'
ARCHIVO_SHP_COLONIAS = 'shp/2023_1_19_A.shp'
CAMPOS = 'pm1.0,pm2.5'

# ------------------ Funciones ------------------ #

def leer_csv(ruta):
    df = pd.read_csv(ruta)
    df = df.dropna(subset=['latitude', 'longitude', 'sensor_index'])
    df['sensor_index'] = df['sensor_index'].astype(int)
    return df

def consultar_sensor(sensor_index):
    url = f'https://api.purpleair.com/v1/sensors/{sensor_index}?fields={CAMPOS}'
    headers = {'X-API-Key': API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json().get("sensor", {})
        return data.get("pm1.0"), data.get("pm2.5")
    return None, None


def clasificar_calidad_aire_pm25(pm25):
    if pm25 is None or np.isnan(pm25):
        return "Sin datos"
    elif pm25 <= 15:
        return "Bueno"
    elif pm25 <= 33:
        return "Aceptable"
    elif pm25 <=79:
        return "Mala"
    elif pm25 <=130:
        return "Muy alta"
    else:
        return "Extremadamente mala"
def clasificar_calidad_aire_pm10(pm10):
    if pm10 is None or np.isnan(pm10):
        return "Sin datos"
    elif pm10 <= 45:
        return "Bueno"
    elif pm10 <= 60:
        return "Aceptable"
    elif pm10 <=132:
        return "Mala"
    elif pm10 <=213:
        return "Muy alta"
    else:
        return "Extremadamente mala"

def crear_geojson(df):
    features = []
    puntos = []
    valores = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    historico_path = "historico.csv"
    datos_historicos = []

    for _, fila in df.iterrows():
        pm1, pm25 = consultar_sensor(fila['sensor_index'])
        if pm1 is not None and pm25 is not None:
            props = {
                "sensor_index": fila['sensor_index'],
                "name": fila.get('name', ''),
                "pm1_0": pm1,
                "pm2_5": pm25,
                "AQ PM 2.5": clasificar_calidad_aire_pm25(pm25),
                "AQ PM 1": clasificar_calidad_aire_pm10(pm1),
                "timestamp": timestamp
            }
            coords = (fila['longitude'], fila['latitude'])
            point = geojson.Point(coords)
            features.append(geojson.Feature(geometry=point, properties=props))
            puntos.append(coords)
            valores.append(pm25)

            # Añadir al historial
            datos_historicos.append({
                "sensor_index": fila['sensor_index'],
                "name": fila.get('name', ''),
                "timestamp": timestamp,
                "pm1_0": pm1,
                "pm2_5": pm25
            })

    # Guardar GeoJSON de sensores
    feature_collection = geojson.FeatureCollection(features)
    with open(SALIDA_GEOJSON_SENSORES, 'w', encoding='utf-8') as f:
        geojson.dump(feature_collection, f, indent=2)

    print(f"GeoJSON de sensores generado: {SALIDA_GEOJSON_SENSORES}")

    # Guardar o actualizar el histórico
    df_historico_nuevo = pd.DataFrame(datos_historicos)

    if os.path.exists(historico_path):
        df_existente = pd.read_csv(historico_path)
        df_total = pd.concat([df_existente, df_historico_nuevo], ignore_index=True)
    else:
        df_total = df_historico_nuevo

    df_total.to_csv(historico_path, index=False, encoding='utf-8')
    print(f"Histórico actualizado: {historico_path}")

    return np.array(puntos), np.array(valores)


    feature_collection = geojson.FeatureCollection(features)
    with open(SALIDA_GEOJSON_SENSORES, 'w', encoding='utf-8') as f:
        geojson.dump(feature_collection, f, indent=2)

    print(f"GeoJSON de sensores generado: {SALIDA_GEOJSON_SENSORES}")
    return np.array(puntos), np.array(valores)

def cargar_datos_colonias_shp(archivo_shp_colonias):
    sf = shapefile.Reader(archivo_shp_colonias, encoding='utf-8')
    colonias = []
    for shape_record in sf.iterShapeRecords():
        geometry = shape(shape_record.shape.__geo_interface__)
        nombre_colonia = shape_record.record[0]
        colonias.append({'nombre': nombre_colonia, 'geometry': geometry})
    return colonias

def interpolar_lineal(punto, triangulo_indices, puntos, valores):
    v0, v1, v2 = puntos[triangulo_indices]
    z0, z1, z2 = valores[triangulo_indices]
    delta1 = v1 - v0
    delta2 = v2 - v0
    delta_p = punto - v0
    try:
        A = np.array([delta1, delta2]).T
        w = np.linalg.solve(A, delta_p)
        b0 = 1 - w[0] - w[1]
        b1, b2 = w
        return b0 * z0 + b1 * z1 + b2 * z2
    except np.linalg.LinAlgError:
        return None

def generar_geojson_colonias(nombre_archivo, colonias_data, puntos_data, valores_puntos, contaminante):
    try:
        tri = Delaunay(puntos_data)
    except ValueError as e:
        print(f"Error en la triangulación de Delaunay: {e}")
        return

    for colonia in colonias_data:
        geom = colonia['geometry']
        puntos_en_colonia = []
        valores_en_colonia = []

        for i, (lon, lat) in enumerate(puntos_data):
            punto = Point(lon, lat)
            if geom.contains(punto):
                puntos_en_colonia.append(punto)
                valores_en_colonia.append(valores_puntos[i])

        if valores_en_colonia:
            colonia['valor_interpolado'] = float(np.mean(valores_en_colonia))
        else:
            centroide = geom.centroid
            punto_centroide = np.array([centroide.x, centroide.y])
            simplex_index = tri.find_simplex(punto_centroide)

            if simplex_index != -1:
                triangulo_indices = tri.simplices[simplex_index]
                valor_interpolado = interpolar_lineal(punto_centroide, triangulo_indices, puntos_data, valores_puntos)
                colonia['valor_interpolado'] = valor_interpolado
            else:
                colonia['valor_interpolado'] = np.nan

    # Crear GeoJSON
    geo_json_data = {
        "type": "FeatureCollection",
        "features": []
    }

    for colonia in colonias_data:
        geom = colonia['geometry']

        if not geom.is_valid or geom.is_empty:
            continue

        try:
            if geom.geom_type == 'Polygon':
                coordinates = [list(geom.exterior.coords)]
                geometry = {
                    "type": "Polygon",
                    "coordinates": coordinates
                }
            elif geom.geom_type == 'MultiPolygon':
                coordinates = []
                for p in geom.geoms:
                    coordinates.append([list(p.exterior.coords)])
                geometry = {
                    "type": "MultiPolygon",
                    "coordinates": coordinates
                }
            else:
                continue
        except Exception:
            continue

        valor_interpolado = colonia.get('valor_interpolado')
        if valor_interpolado is not None and not np.isnan(valor_interpolado):
            valor_export = round(float(valor_interpolado), 2)
        else:
            valor_export = None

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "nombre": colonia['nombre'],
                "valor_interpolado": valor_export,
                "AQ": clasificar_calidad_aire_pm25(valor_export) if contaminante == 'pm2_5' else clasificar_calidad_aire_pm10(valor_export)
            }
        }

        geo_json_data["features"].append(feature)

    with open(nombre_archivo, "w", encoding="utf-8") as f:
        json.dump(geo_json_data, f, ensure_ascii=False, indent=2)

    print(f"GeoJSON generado para {contaminante}: {nombre_archivo}")


# ------------------ Ejecución Principal ------------------ #

if __name__ == '__main__':
    # Sensore
    df_sensores = leer_csv(CSV_FILE)
    puntos_data, valores_puntos = crear_geojson(df_sensores)
    #Colonias
    colonias_data = cargar_datos_colonias_shp(ARCHIVO_SHP_COLONIAS)
    generar_geojson_colonias(SALIDA_GEOJSON_COLONIAS_PM25, colonias_data, puntos_data, valores_puntos, 'pm2_5')
    generar_geojson_colonias(SALIDA_GEOJSON_COLONIAS_PM10, colonias_data, puntos_data, valores_puntos, 'pm1_0')
