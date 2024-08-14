# raster_analysis



gdalwarp 2yr_flood_extent.tif 2yr_flood_extent-4326.tif -s_srs EPSG:3857 -t_srs EPSG:4326

gdal_translate -of COG 2yr_flood_extent-4326.tif 2yr_flood_extent-cog.tif



sudo -u postgres osm2pgsql --create -d practice nepal-latest.osm.pbf -U postgres


-- Create the buildings table
CREATE TABLE buildings (
  id SERIAL PRIMARY KEY,
  osm_id BIGINT,
  building VARCHAR,
  geometry GEOMETRY(Polygon, 4326)
);

-- Insert data from planet_osm_polygon into buildings table
INSERT INTO buildings (osm_id, building, geometry)
SELECT osm_id, building, ST_Transform(way, 4326)  -- Transform if needed
FROM planet_osm_polygon
WHERE building IS NOT NULL;


to generate the h6 cells

ALTER TABLE buildings ADD COLUMN h3_index h3index GENERATED ALWAYS AS (h3_lat_lng_to_cell(ST_Centroid(geometry), 8)) STORED;

to select the sample data

SELECT *
FROM buildings
LIMIT 10;




python3 -m venv env

source env/bin/activate

pip install "fastapi[standard]"

uvicorn main:app --reload


pip install h3 h3ronpy rasterio asyncio asyncpg aiohttp
