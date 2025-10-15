import os
import json
import pandas as pd
import numpy as np
import requests
import geojson
import shapefile
from shapely.geometry import shape, Point
from scipy.spatial import Delaunay
from datetime import datetime, timezone  # Importar timezone

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
    # Se asegura que existan lat/lon y sensor_index
    df = df.dropna(subset=['latitude', 'longitude', 'sensor_index'])
    df['sensor_index'] = df['sensor_index'].astype(int)
    return df

def consultar_sensor(sensor_index):
    url = f'https://api.purpleair.com/v1/sensors/{sensor_index}?fields={CAMPOS}'
    headers = {'X-API-Key': API_KEY} if API_KEY else {}
    try:
        response = requests.get(url, headers=headers, timeout=10)
    except Exception:
        return None, None
    if response.status_code == 200:
        data = response.json().get("sensor", {})
        # Devuelve pm1.0 y pm2.5 (o None si no vienen)
        return data.get("pm1.0"), data.get("pm2.5")
    return None, None

def clasificar_calidad_aire_pm25(pm25):
    if pm25 is None or (isinstance(pm25, float) and np.isnan(pm25)):
        return "Sin datos"
    try:
        pm25 = float(pm25)
    except Exception:
        return "Sin datos"
    if pm25 <= 15:
        return "Bueno"
    elif pm25 <= 33:
        return "Aceptable"
    elif pm25 <= 79:
        return "Mala"
    elif pm25 <= 130:
        return "Muy alta"
    else:
        return "Extremadamente mala"

def clasificar_calidad_aire_pm10(pm10):
    if pm10 is None or (isinstance(pm10, float) and np.isnan(pm10)):
        return "Sin datos"
    try:
        pm10 = float(pm10)
    except Exception:
        return "Sin datos"
    if pm10 <= 45:
        return "Bueno"
    elif pm10 <= 60:
        return "Aceptable"
    elif pm10 <= 132:
        return "Mala"
    elif pm10 <= 213:
        return "Muy alta"
    else:
        return "Extremadamente mala"

# MODIFICACIÓN 1: Se añade el parámetro 'timestamp' a la función
def crear_geojson(df, timestamp):
    features = []
    puntos = []
    valores_pm25 = []
    valores_pm10 = []

    historico_path = "historico.csv"
    datos_historicos = []

    for _, fila in df.iterrows():
        # consultar_sensor devuelve (pm1.0, pm2.5)
        pm10, pm25 = consultar_sensor(fila['sensor_index'])
        # Nota: en tu código original pm10 toma pm1.0 y pm25 toma pm2.5.
        # Si esperas PM10 real, asegúrate de que la API lo provea.
        if pm10 is not None and pm25 is not None:
            props = {
                "sensor_index": int(fila['sensor_index']),
                "name": fila.get('name', ''),
                "pm1_0": pm10,
                "pm2_5": pm25,
                "AQ PM 2.5": clasificar_calidad_aire_pm25(pm25),
                "AQ PM 10": clasificar_calidad_aire_pm10(pm10),
                "timestamp": timestamp  # Se usa el timestamp recibido
            }
            coords = (float(fila['longitude']), float(fila['latitude']))
            point = geojson.Point(coords)
            features.append(geojson.Feature(geometry=point, properties=props))
            puntos.append(coords)
            # valores que se interpolarán
            valores_pm25.append(float(pm25))
            valores_pm10.append(float(pm10))

            # Añadir al historial
            datos_historicos.append({
                "sensor_index": int(fila['sensor_index']),
                "name": fila.get('name', ''),
                "timestamp": timestamp,  # Se usa el timestamp recibido
                "pm1_0": pm10,
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
        try:
            df_existente = pd.read_csv(historico_path)
            df_total = pd.concat([df_existente, df_historico_nuevo], ignore_index=True)
        except Exception:
            df_total = df_historico_nuevo
    else:
        df_total = df_historico_nuevo
    df_total.to_csv(historico_path, index=False, encoding='utf-8')
    print(f"Histórico actualizado: {historico_path}")

    return np.array(puntos), np.array(valores_pm25), np.array(valores_pm10)

def cargar_datos_colonias_shp(archivo_shp_colonias):
    sf = shapefile.Reader(archivo_shp_colonias, encoding='utf-8')
    colonias = []
    for shape_record in sf.iterShapeRecords():
        geometry = shape(shape_record.shape.__geo_interface__)
        # Se asume que el nombre está en el primer campo del registro
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

# MODIFICACIÓN 2: Se añade el parámetro 'timestamp' a la función
def generar_geojson_colonias(nombre_archivo, colonias_data, puntos_data, valores_puntos, contaminante, timestamp):
    try:
        tri = Delaunay(puntos_data)
    except Exception as e:
        print(f"Error en la triangulación de Delaunay: {e}")
        tri = None

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
            if tri is None:
                colonia['valor_interpolado'] = np.nan
                continue

            centroide = geom.centroid
            punto_centroide = np.array([centroide.x, centroide.y])
            simplex_index = tri.find_simplex(punto_centroide)

            if simplex_index != -1:
                triangulo_indices = tri.simplices[simplex_index]
                valor_interpolado = interpolar_lineal(punto_centroide, triangulo_indices, puntos_data, valores_puntos)
                colonia['valor_interpolado'] = valor_interpolado
            else:
                colonia['valor_interpolado'] = np.nan

    # Crear GeoJSON con metadatos
    geo_json_data = {
        "type": "FeatureCollection",
        "metadata": {
            "ultima_ejecucion_utc": timestamp
        },
        "features": []
    }

    for colonia in colonias_data:
        geom = colonia['geometry']
        if not geom.is_valid or geom.is_empty:
            continue

        try:
            if geom.geom_type == 'Polygon':
                coordinates = [list(geom.exterior.coords)]
                geometry = {"type": "Polygon", "coordinates": coordinates}
            elif geom.geom_type == 'MultiPolygon':
                coordinates = [[list(p.exterior.coords)] for p in geom.geoms]
                geometry = {"type": "MultiPolygon", "coordinates": coordinates}
            else:
                continue
        except Exception:
            continue

        valor_interpolado = colonia.get('valor_interpolado')
        valor_export = round(float(valor_interpolado), 2) if valor_interpolado is not None and not np.isnan(valor_interpolado) else None

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
    # MODIFICACIÓN 3: Se genera un único timestamp para toda la ejecución
    timestamp_ejecucion = datetime.now(timezone.utc).isoformat()
    print(f"Iniciando ejecución: {timestamp_ejecucion}")

    # Sensores
    df_sensores = leer_csv(CSV_FILE)
    # Se pasa el timestamp a la función y se reciben los valores de PM2.5 y PM10 por separado
    puntos_data, valores_pm25, valores_pm10 = crear_geojson(df_sensores, timestamp_ejecucion)

    # Colonias
    colonias_data_pm25 = cargar_datos_colonias_shp(ARCHIVO_SHP_COLONIAS)
    colonias_data_pm10 = cargar_datos_colonias_shp(ARCHIVO_SHP_COLONIAS)  # Cargar una copia para PM10

    # Se pasa el timestamp a las funciones de generación de GeoJSON
    if puntos_data.size == 0:
        print("No hay puntos válidos para interpolación. Se generaron los GeoJSON de sensores e histórico pero no se hará la interpolación de colonias.")
    else:
        generar_geojson_colonias(SALIDA_GEOJSON_COLONIAS_PM25, colonias_data_pm25, puntos_data, valores_pm25, 'pm2_5', timestamp_ejecucion)
        generar_geojson_colonias(SALIDA_GEOJSON_COLONIAS_PM10, colonias_data_pm10, puntos_data, valores_pm10, 'pm10', timestamp_ejecucion)

    # ------------------ Generar historico_combinado.geojson ------------------ #
    try:
        import geopandas as gpd
    except Exception as e:
        gpd = None
        print("Aviso: geopandas no está disponible. Instala geopandas para generar 'historico_combinado.geojson'. Error:", e)

    try:
        if gpd is None:
            raise ImportError("geopandas no disponible")

        historico_path = "historico.csv"
        # Leer histórico (que se actualizó en crear_geojson) y sensores_detectados
        if os.path.exists(historico_path):
            df_historico = pd.read_csv(historico_path, encoding='utf-8')
        else:
            df_historico = pd.DataFrame()  # vacío si no existe

        if os.path.exists(CSV_FILE):
            df_sensores_detectados = pd.read_csv(CSV_FILE, encoding='utf-8')
        else:
            df_sensores_detectados = pd.DataFrame()

        if df_historico.empty:
            print("El archivo histórico está vacío o no existe; no se generará 'historico_combinado.geojson'.")
        else:
            # Intentar unir por 'name'
            if 'name' not in df_historico.columns:
                print("Advertencia: 'name' no existe en historico.csv; no se puede unir. Se copiará el histórico sin geometría.")
                df_combinado = df_historico.copy()
            else:
                # Tomar las columnas de lat/lon de sensores_detectados (si existen)
                cols_needed = []
                if 'latitude' in df_sensores_detectados.columns and 'longitude' in df_sensores_detectados.columns:
                    cols_needed = ['name', 'latitude', 'longitude']
                elif 'lat' in df_sensores_detectados.columns and 'lon' in df_sensores_detectados.columns:
                    df_sensores_detectados = df_sensores_detectados.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
                    cols_needed = ['name', 'latitude', 'longitude']

                if cols_needed:
                    df_combinado = df_historico.merge(df_sensores_detectados[cols_needed], on='name', how='left')
                else:
                    print("Advertencia: sensores_detectados.csv no tiene columnas 'latitude'/'longitude'. Se copiará el histórico sin geometría.")
                    df_combinado = df_historico.copy()

            # Limpiar filas sin coordenadas antes de crear geometría
            if 'longitude' in df_combinado.columns and 'latitude' in df_combinado.columns:
                df_combinado_geo = df_combinado.dropna(subset=['latitude', 'longitude']).copy()
                if not df_combinado_geo.empty:
                    # Asegurar tipos numéricos
                    df_combinado_geo['longitude'] = df_combinado_geo['longitude'].astype(float)
                    df_combinado_geo['latitude'] = df_combinado_geo['latitude'].astype(float)

                    gdf = gpd.GeoDataFrame(
                        df_combinado_geo,
                        geometry=gpd.points_from_xy(df_combinado_geo.longitude, df_combinado_geo.latitude),
                        crs="EPSG:4326"
                    )
                    # Guardar GeoJSON con todas las columnas originales + geometry
                    salida_comb = "historico_combinado.geojson"
                    gdf.to_file(salida_comb, driver="GeoJSON")
                    print(f"GeoJSON combinado generado: {salida_comb}")
                else:
                    print("No se encontraron filas con latitud/longitud en el histórico combinado; no se creó el GeoJSON combinado.")
            else:
                print("No hay columnas de lat/lon para crear geometría en el histórico combinado; se omitió la creación del GeoJSON combinado.")
    except Exception as e:
        print(f"⚠️ Error al generar historico_combinado.geojson: {e}")

    print("Proceso completado.")
