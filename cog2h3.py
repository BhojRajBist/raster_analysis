import argparse
import asyncio
import logging
import os
import time
from urllib.parse import urlparse

import aiohttp
import asyncpg
import h3
import numpy as np
import pandas as pd
import rasterio
from h3ronpy.arrow.raster import nearest_h3_resolution, raster_to_dataframe
from rasterio.enums import Resampling
from sklearn.preprocessing import MinMaxScaler

from __version__ import __version__


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATABASE_URL = "postgres://postgres:postgres@localhost/practice"
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
## setup static dir for cog downloads if required
STATIC_DIR = os.getenv("STATIC_DIR", "static")


async def download_cog(cog_url_or_path: str) -> str:
    """Checks if the supplied string is a file path or URL.
    Downloads the COG to static dir if needed.

    Args:
        cog_url_or_path (str): URL or local file path to check/download.

    Raises:
        Exception: Raised if download fails.

    Returns:
        str: File path of the COG.
    """
    if os.path.isfile(cog_url_or_path):
        logging.info(f"COG file already exists at {cog_url_or_path}")
        return cog_url_or_path

    parsed_url = urlparse(cog_url_or_path)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise Exception(f"Invalid URL or file path: {cog_url_or_path}")

    cog_file_name = os.path.basename(parsed_url.path)
    os.makedirs(STATIC_DIR, exist_ok=True)
    file_path = os.path.join(STATIC_DIR, cog_file_name)

    if os.path.exists(file_path):
        logging.info(f"COG file already exists in static dir: {file_path}")
        return file_path

    logging.info(f"Downloading COG from {cog_url_or_path}")
    async with aiohttp.ClientSession() as session:
        async with session.get(cog_url_or_path) as response:
            if response.status != 200:
                logging.error(f"Failed to download COG from {cog_url_or_path}")
                raise Exception(f"Failed to download COG from {cog_url_or_path}")
            with open(file_path, "wb") as tmp_file:
                tmp_file.write(await response.read())
                logging.info(f"Downloaded COG to {file_path}")
                return file_path


async def create_or_replace_table_pandas(df: pd.DataFrame, table_name: str, db_url: str):
    logging.info(f"Creating or replacing table {table_name} in database")
    start_time = time.time()
    columns = df.columns.tolist()

    conn = await asyncpg.connect(dsn=db_url)

    create_table_query = f"""
    CREATE TABLE {table_name} (
        h3_ix h3index PRIMARY KEY,
        {', '.join([f"{col} FLOAT" for col in columns if col != 'h3_ix'])}
    )
    """

    insert_columns = ", ".join(columns)
    insert_placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
    insert_query = f"INSERT INTO {table_name} ({insert_columns}) VALUES ({insert_placeholders})"

    await conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    await conn.execute(create_table_query)

    data = [tuple(row) for row in df.to_numpy()]

    # h3_index_value_pairs = list(zip(h3_indexes, values))
    await conn.executemany(
        insert_query,
        data,
    )

    await conn.close()

    end_time = time.time()
    logging.info(f"Table {table_name} created or updated successfully in {end_time - start_time:.2f} seconds.")


def convert_h3_indices_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """Convert H3 indices to hex strings in a streaming manner to handle large datasets."""
    logging.info("Converting H3 indices to hex strings")
    df["h3_ix"] = df["cell"].apply(h3.h3_to_string)
    df = df.drop(columns=["cell"])

    return df


def get_edge_length(res, unit="km"):
    """Gets edge length of constant h3 cells using resolution"""

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


async def process_raster(cog_url: str, table_name: str, h3_res: int, sample_by: str, preserve_range: bool, multiband: bool):
    """Resamples and generates h3 value for raster"""
    cog_file_path = await download_cog(cog_url)
    raster_time = time.time()

    logging.info(f"Processing raster file: {cog_file_path}")
    with rasterio.open(cog_file_path) as src:
        band_count = src.count
        if not multiband:
            band_count = 1

        transform = src.transform
        bands_data = []

        for band in range(1, band_count + 1):
            grayscale = src.read(band)
            min_value = np.min(grayscale)
            max_value = np.max(grayscale)
            native_h3_res = nearest_h3_resolution(grayscale.shape, src.transform, search_mode="smaller_than_pixel")
            logging.info(f"Determined Min fitting H3 resolution for band {band}: {native_h3_res}")

            if h3_res > native_h3_res:
                logging.warn(
                    f"Supplied res {h3_res} is higher than native resolution, Upscaling raster is not supported yet, hence falling back to {native_h3_res}"
                )
                h3_res = native_h3_res

            if h3_res < native_h3_res:
                logging.info(f"Resampling original raster to: {get_edge_length(h3_res-1, unit='m')}m")
                scale_factor = src.res[0] / (get_edge_length(h3_res - 1, unit="m") / 111320)
                data = src.read(
                    band,
                    out_shape=(
                        int(src.height * scale_factor),
                        int(src.width * scale_factor),
                    ),
                    resampling=Resampling[sample_by],
                )
                transform = src.transform * src.transform.scale((src.width / data.shape[-1]), (src.height / data.shape[-2]))
                grayscale = data
                logging.info(f"Resampling Done for band {band}")

                if preserve_range:
                    scaler = MinMaxScaler(feature_range=(min_value, max_value))
                    grayscale = scaler.fit_transform(grayscale.reshape(-1, 1)).reshape(grayscale.shape)

                native_h3_res = nearest_h3_resolution(grayscale.shape, transform, search_mode="smaller_than_pixel")
                logging.info(f"New Native H3 resolution for band {band}: {native_h3_res}")

            grayscale_h3_df = raster_to_dataframe(
                grayscale,
                transform,
                native_h3_res,
                nodata_value=0,
                compact=False,
            )
            logging.info(f"Calculation done for res:{native_h3_res} band:{band}")
            grayscale_h3_df_pandas = grayscale_h3_df.to_pandas()
            grayscale_h3_df_pandas = grayscale_h3_df_pandas.rename(columns={"value": f"band{band}"})
            bands_data.append(grayscale_h3_df_pandas)

        # Merge all bands data
        if multiband:
            result_df = bands_data[0]
            for df in bands_data[1:]:
                result_df = result_df.merge(df, on="cell", how="outer")
        else:
            result_df = bands_data[0]

        result_h3_merged_df = convert_h3_indices_pandas(result_df)

        logging.info(f"Overall raster calculation done in {int(time.time()-raster_time)} seconds")

        await create_or_replace_table_pandas(result_h3_merged_df, table_name, DATABASE_URL)


def main():
    """Iron Man Main function"""
    parser = argparse.ArgumentParser(description="Process a Cloud Optimized GeoTIFF and upload data to PostgreSQL.")
    parser.add_argument(
        "--cog",
        type=str,
        required=True,
        help="URL of the Cloud Optimized GeoTIFF",  ### IMP : This should be in wgs84
    )
    parser.add_argument("--table", type=str, required=True, help="Name of the database table")
    parser.add_argument("--res", type=int, default=8, help="H3 resolution level")
    parser.add_argument(
        "--preserve_range",
        action="store_true",
        help="Preserve range of raster while resampling",
    )
    parser.add_argument(
        "--multiband",
        action="store_true",
        help="Enable this if you raster contains multiple band and you wanna compute value for each of them",
    )
    parser.add_argument(
        "--sample_by",
        type=str,
        default="bilinear",
        choices=resampling_methods,
        help="Raster Resampling Method",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    logging.info("Starting processing")
    asyncio.run(process_raster(args.cog, args.table, args.res, args.sample_by, args.preserve_range, args.multiband))
    logging.info("Processing completed")


if __name__ == "__main__":
    main()