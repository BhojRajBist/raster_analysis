import aiohttp
import asyncio
import logging
import os
import time
import h3
import numpy as np
import pyarrow as pa
import rasterio
from rasterio.enums import Resampling
from h3ronpy.arrow.raster import nearest_h3_resolution, raster_to_dataframe
from models import create_or_replace_table_arrow, convert_h3_indices_arrow

DATABASE_URL = 'postgresql://postgres:postgres@localhost/practice'

async def download_cog(cog_url: str) -> str:
    """Downloads COG to file dir if not exists."""
    cog_file_name = os.path.basename(cog_url)
    file_path = os.path.join(STATIC_DIR, cog_file_name)

    if os.path.exists(file_path):
        logging.info(f"COG file already exists: {file_path}")
        return file_path

    logging.info(f"Downloading COG from {cog_url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(cog_url) as response:
            if response.status != 200:
                logging.error(f"Failed to download COG from {cog_url}")
                raise Exception(f"Failed to download COG from {cog_url}")
            with open(file_path, "wb") as tmp_file:
                tmp_file.write(await response.read())
                logging.info(f"Downloaded COG to {file_path}")
                return file_path

def get_edge_length(res, unit="km"):
    """Gets edge length of constant H3 cells using resolution."""
    edge_lengths_km = [
        1281.256011, 483.0568391, 182.5129565, 68.97922179, 26.07175968,
        9.854090990, 3.724532667, 1.406475763, 0.531414010, 0.200786148,
        0.075863783, 0.028663897, 0.010830188, 0.004092010, 0.001546100,
        0.000584169
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

async def process_raster(cog_url: str, table_name: str, h3_res: int, sample_by: str):
    """Resamples and generates H3 value for raster."""
    cog_file_path = await download_cog(cog_url)
    logging.info(f"Processing raster file: {cog_file_path}")
    with rasterio.open(cog_file_path) as src:
        grayscale = src.read(1)
        transform = src.transform

        native_h3_res = nearest_h3_resolution(
            grayscale.shape, src.transform, search_mode="smaller_than_pixel"
        )
        logging.info(f"Determined Min fitting H3 resolution: {native_h3_res}")

        if h3_res > native_h3_res:
            logging.warning(
                f"Supplied res {h3_res} is higher than native resolution, "
                "Upscaling raster is not supported yet, hence falling back to {native_h3_res}"
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
            nodata_value=None,
            compact=False,
        )
        grayscale_h3_df = convert_h3_indices_arrow(grayscale_h3_df)
        logging.info(f"Raster calculation done")
        await create_or_replace_table_arrow(grayscale_h3_df, table_name, DATABASE_URL)
