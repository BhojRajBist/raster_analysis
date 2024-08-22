from fastapi import FastAPI, File, UploadFile, Form, HTTPException,Depends, Query
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
from fastapi.middleware.cors import CORSMiddleware

# code added to remove the raster with no data
from rasterio.windows import get_data_window
from rasterio.windows import transform as trfs
from sklearn.preprocessing import MinMaxScaler



import asyncpg
from asyncache import cached
from cachetools import TTLCache
from fastapi.responses import Response


from sqlalchemy import create_engine, text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Change this to specific domains in production.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

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

            nodata_value = src.nodata   # this code is not working I think so
            print(nodata_value)
            if nodata_value is not None:
                grayscale = np.where(grayscale == nodata_value, 0, grayscale)

            native_h3_res = nearest_h3_resolution(
                grayscale.shape, transform, search_mode="smaller_than_pixel"
            )
            logging.info(f"New Native H3 resolution: {native_h3_res}")

                    # Filter out NaN values
        grayscale = np.nan_to_num(grayscale, nan=0)

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



# to visulaize the data as MVT tiles
# Create a cache with a maximum of 1000 items and a 1-hour TTL
# DATABASE_URL = os.getenv(
#     "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
# )

TABLE = {
    'table': os.getenv('TILE_TABLE_NAME', 'two_year_flood'),
    'srid': os.getenv('TILE_TABLE_SRID', '4326'),
    'h3inxColumn': os.getenv('TILE_TABLE_H3INX_COLUMN', 'h3_ix'),
    'h3inxRes': os.getenv('TILE_TABLE_H3INX_RESOLUTION', 8),
    'attrColumns': os.getenv('TILE_TABLE_ATTR_COLUMNS', 'cell_value')
}


cache = TTLCache(maxsize=1000, ttl=3600)

async def get_db_pool():
    return await asyncpg.create_pool(DATABASE_URL)

@cached(cache)
async def get_tile(zoom: int, x: int, y: int, pool):
    # async with pool.acquire() as conn:
    #     env = tile_to_envelope(zoom, x, y)
    #     sql = envelope_to_sql(env)
    #     logging.debug(f"SQL Query: {sql}")  # Print the SQL query
    #     pbf = await conn.fetchval(sql)
    #     logging.debug(f"PBF data received. Size: {len(pbf)} bytes")
    #     return await conn.fetchval(sql)
    async with pool.acquire() as conn:
        env = tile_to_envelope(zoom, x, y)
        sql = envelope_to_sql(env)
        logging.debug(f"SQL Query: {sql}")  # Print the SQL query
        
        try:
            pbf = await conn.fetchval(sql)
            if not pbf:
                raise Exception("No PBF data returned from the query")
            logging.debug(f"PBF data received. Size: {len(pbf)} bytes")
            return pbf
        except Exception as e:
            logging.error(f"Error fetching tile: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
def tile_to_envelope(zoom: int, x: int, y: int):
    world_merc_max = 20037508.3427892
    world_merc_min = -world_merc_max
    world_merc_size = world_merc_max - world_merc_min
    world_tile_size = 2 ** zoom
    tile_merc_size = world_merc_size / world_tile_size
    
    env = {
        'xmin': world_merc_min + tile_merc_size * x,
        'xmax': world_merc_min + tile_merc_size * (x + 1),
        'ymin': world_merc_max - tile_merc_size * (y + 1),
        'ymax': world_merc_max - tile_merc_size * y
    }
    return env

def envelope_to_bounds_sql(env):
    DENSIFY_FACTOR = 4
    env['segSize'] = (env['xmax'] - env['xmin']) / DENSIFY_FACTOR
    sql_tmpl = 'ST_Segmentize(ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, 3857), {segSize})'
    return sql_tmpl.format(**env)

def envelope_to_sql(env):
    tbl = TABLE.copy()
    tbl['env'] = envelope_to_bounds_sql(env)
    sql_tmpl = """
        WITH 
         bounds AS (
            SELECT {env} AS geom, 
                   {env}::box2d AS b2d
        ),
        mvtgeom AS (
            SELECT ST_AsMVTGeom(ST_Transform(h3_cell_to_boundary_geometry(t.{h3inxColumn}), 3857), bounds.b2d) AS geom, 
                   {attrColumns}
            FROM {table} t, bounds
            WHERE {h3inxColumn} = ANY (get_h3_indexes(ST_Transform(bounds.geom, {srid}),{h3inxRes}))
        ) 
        SELECT ST_AsMVT(mvtgeom.*) FROM mvtgeom
    """
    return sql_tmpl.format(**tbl)

@app.on_event("startup")
async def startup_event():
    app.state.pool = await get_db_pool()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.pool.close()

@app.get("/{zoom}/{x}/{y}.{format}")
async def get_mvt_tile(zoom: int, x: int, y: int, format: str):
    if format not in ['pbf', 'mvt']:
        raise HTTPException(status_code=400, detail="Invalid format. Use 'pbf' or 'mvt'.")
    
    tile_size = 2 ** zoom
    if x < 0 or y < 0 or x >= tile_size or y >= tile_size:
        raise HTTPException(status_code=400, detail="Invalid tile coordinates.")

    try:
        pbf = await get_tile(zoom, x, y, app.state.pool)
        return Response(content=pbf, media_type="application/vnd.mapbox-vector-tile")
    except Exception as e:
        # raise e
        raise HTTPException(status_code=500, detail=str(e))




#get the ward from the datbase
engine = create_engine(DATABASE_URL)

def get_distinct_values(column_name: str, filter_conditions: dict = None):
    query = f'SELECT DISTINCT "{column_name}" FROM wards'
    conditions = []
    params = {}
    if filter_conditions:
        for column, value in filter_conditions.items():
            if value is not None:
                conditions.append(f'"{column}" = :{column}')
                params[column] = value
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += f' ORDER BY "{column_name}"'
    
    with engine.connect() as connection:
        result = connection.execute(text(query), params).fetchall()
        return [row[0] for row in result]

# Endpoint to get the ward data with dropdown options
@app.get("/ward-data/")
async def get_ward_data(
    state_code: int = Query(None, description="Select State Code"),
    district: str = Query(None, description="Select District"),
    municipality: str = Query(None, description="Select Municipality"),
    ward_number: int = Query(None, description="Select Ward Number"),
):

    if state_code is None and district is None and municipality is None and ward_number is None:
        state_codes = get_distinct_values("STATE_CODE")
        districts = get_distinct_values("DISTRICT")
        municipalities = get_distinct_values("GaPa_NaPa")
        wards = get_distinct_values("NEW_WARD_N")

        return {
            "state_codes": state_codes,
            "districts": districts,
            "municipalities": municipalities,
            "wards": wards,
        }
    elif state_code and district and municipality and ward_number:
        query = text("""
            SELECT ST_AsGeoJSON(geom) as geojson
            FROM wards
            WHERE "STATE_CODE" = :state_code
            AND "DISTRICT" = :district
            AND "GaPa_NaPa" = :municipality
            AND "NEW_WARD_N" = :ward_number
            LIMIT 1;
        """)

        with engine.connect() as connection:
            result = connection.execute(query, {
                "state_code": state_code,
                "district": district,
                "municipality": municipality,
                "ward_number": ward_number
            }).mappings().fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Ward not found")

        geojson = result['geojson']
        return json.loads(geojson)

    else:
        filter_conditions = {
            "STATE_CODE": state_code,
            "DISTRICT": district,
            "GaPa_NaPa": municipality
        }

        if state_code is not None and district is None:
            districts = get_distinct_values("DISTRICT", filter_conditions)
            return {"districts": districts}

        elif district is not None and municipality is None:
            municipalities = get_distinct_values("GaPa_NaPa", filter_conditions)
            return {"municipalities": municipalities}

        elif municipality is not None and ward_number is None:
            wards = get_distinct_values("NEW_WARD_N", filter_conditions)
            return {"wards": wards}

        raise HTTPException(status_code=400, detail="Invalid parameter combination")
