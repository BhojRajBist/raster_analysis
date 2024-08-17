from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
import io
import logging
import aiohttp
import asyncpg
import numpy as np
import rasterio
from rasterio.enums import Resampling
from h3ronpy.arrow.raster import nearest_h3_resolution, raster_to_dataframe
import h3
import os
import pandas as pd
import json

# code added to remove the raster with no data
from rasterio.windows import get_data_window
from rasterio.windows import transform as trfs
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/practice"

resampling_methods = [
    "nearest",
    "bilinear",
    "cubic",
    "cubic_spline",
    "lanczos",
    "average",
    "mode",
    "gauss",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "sum",
    "rms",
]

STATIC_DIR = "static"
os.makedirs(STATIC_DIR, exist_ok=True)

class ProcessRequest(BaseModel):
    table_name: str
    h3_res: int
    sample_by: str

async def download_cog(file: UploadFile) -> str:
    cog_file_name = file.filename
    file_path = os.path.join(STATIC_DIR, cog_file_name)
    
    if os.path.exists(file_path):
        logging.info(f"COG file already exists: {file_path}")
        return file_path
    
    logging.info(f"Saving COG file to {file_path}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    logging.info(f"Saved COG to {file_path}")
    return file_path

async def create_or_replace_table_pandas(df: pd.DataFrame, table_name: str, db_url: str):
    logging.info(f"Creating or replacing table {table_name} in database")
    conn = await asyncpg.connect(dsn=db_url)

    await conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    await conn.execute(
        f"""
        CREATE TABLE {table_name} (
            h3_ix h3index PRIMARY KEY,
            cell_value FLOAT
        )
    """
    )

    h3_indexes = df["h3_ix"].tolist()
    values = df["value"].tolist()

    h3_index_value_pairs = list(zip(h3_indexes, values))
    await conn.executemany(
        f"INSERT INTO {table_name} (h3_ix, cell_value) VALUES ($1, $2)",
        h3_index_value_pairs,
    )

    await conn.close()
    logging.info(f"Table {table_name} created or updated successfully")

def convert_h3_indices_pandas(df: pd.DataFrame) -> pd.DataFrame:
    df["h3_ix"] = df["cell"].apply(h3.h3_to_string)
    df = df.drop(columns=["cell"])
    return df

def get_edge_length(res, unit="km"):
    edge_lengths_km = [
        1281.256011, 
        483.0568391,
        182.5129565, 
        68.97922179, 
        26.07175968,
        9.854090990, 
        3.724532667, 
        1.406475763, 
        0.531414010, 
        0.200786148,
        0.075863783, 
        0.028663897, 
        0.010830188, 
        0.004092010, 
        0.001546100,
        0.000584169,
    ]

    if res < 0 or res >= len(edge_lengths_km):
        raise ValueError("Invalid resolution. It should be between 0 and 15.")

    edge_length_km = edge_lengths_km[res]

    if unit == "km":
        return edge_length_km
    elif unit == "m":
        return edge_length_km * 1000
    else:
        raise ValueError("Invalid unit. Use 'km' for kilometers or 'm' for meters.")

async def process_raster(file: UploadFile, table_name: str, h3_res: int, sample_by: str, preserve_range: bool= True):
    cog_file_path = await download_cog(file)
    logging.info(f"Processing raster file: {cog_file_path}")

    with rasterio.open(cog_file_path) as src:
        grayscale = src.read(1)
        transform = src.transform
        min_value = np.min(grayscale)
        max_value = np.max(grayscale)

        native_h3_res = nearest_h3_resolution(
            grayscale.shape, src.transform, search_mode="smaller_than_pixel"
        )
        logging.info(f"Determined Min fitting H3 resolution: {native_h3_res}")

        if h3_res > native_h3_res:
            logging.warning(
                f"Supplied res {h3_res} is higher than native resolution, falling back to {native_h3_res}"
            )
            h3_res = native_h3_res

        if h3_res < native_h3_res:
            logging.info(
                f"Resampling original raster to : {get_edge_length(h3_res-1, unit='m')}m"
            )
            scale_factor = src.res[0] / (get_edge_length(h3_res - 1, unit="m") / 111320)
            data = src.read(
                out_shape=(
                    src.count,
                    int(src.height * scale_factor),
                    int(src.width * scale_factor),
                ),
                resampling=Resampling[sample_by],
            )
            transform = src.transform * src.transform.scale(
                (src.width / data.shape[-1]), (src.height / data.shape[-2])
            )
            grayscale = data[0]
            logging.info("Resampling Done")

            if preserve_range:
                scaler = MinMaxScaler(feature_range=(min_value, max_value))
                grayscale = scaler.fit_transform(grayscale.reshape(-1, 1)).reshape(
                grayscale.shape
                )

            nodata_value = src.nodata
            if nodata_value is not None:
                grayscale = np.where(grayscale == nodata_value, 0, grayscale)

            native_h3_res = nearest_h3_resolution(
                grayscale.shape, transform, search_mode="smaller_than_pixel"
            )
            logging.info(f"New Native H3 resolution: {native_h3_res}")

        grayscale_h3_df = raster_to_dataframe(
            grayscale,
            transform,
            native_h3_res,
            nodata_value=0,
            compact=True,
        )
        grayscale_h3_df_pandas = grayscale_h3_df.to_pandas()
        grayscale_h3_df = convert_h3_indices_pandas(grayscale_h3_df_pandas)
        await create_or_replace_table_pandas(
            grayscale_h3_df_pandas, table_name, DATABASE_URL
        )

@app.post("/process/")
async def process_file(
    file: UploadFile = File(...),
    table_name: str = Form(...),
    h3_res: int = Form(...),
    sample_by: str = Form(...),
):
    if sample_by not in resampling_methods:
        raise HTTPException(status_code=400, detail="Invalid resampling method")
    
    await process_raster(file, table_name, h3_res, sample_by)
    return {"status": "success"}



async def fetch_h3_data(table_name: str, db_url: str) -> dict:
    conn = await asyncpg.connect(dsn=db_url)
    query = f"SELECT h3_ix, cell_value FROM {table_name};"
    rows = await conn.fetch(query)
    await conn.close()

    features = []
    for row in rows:
        h3_index = row["h3_ix"]
        cell_value = row["cell_value"]
        polygon = h3.h3_to_geo_boundary(h3_index, geo_json=True)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            },
            "properties": {
                "cell_value": cell_value
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features
    }

@app.get("/visualize/{table_name}")
async def visualize_h3_data(table_name: str):
    try:
        geojson_data = await fetch_h3_data(table_name, DATABASE_URL)
        return geojson_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))