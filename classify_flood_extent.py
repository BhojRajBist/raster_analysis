import rasterio
import numpy as np
from rasterio.features import shapes
from rasterio.warp import transform_geom
from typing import List, Dict
from sqlalchemy.orm import Session
from models import FloodZone
import logging

def classify_flood_extent(file_path: str) -> List[Dict]:
    with rasterio.open(file_path) as src:
        raster = src.read(1)
        transform = src.transform

        # Mask out pixels below or equal to 0 
        raster = np.where(raster <= 0, np.nan, raster)

        # Classification thresholds based on PDRA
        shallow_threshold = 0.5
        moderate_threshold = 1.5
        deep_threshold = 3.0

        # Create an empty mask with the same shape as the raster
        classification = np.zeros_like(raster, dtype=np.uint8)

        # Assign classification values based on flood depth thresholds
        classification[raster > deep_threshold] = 3
        classification[(raster > moderate_threshold) & (raster <= deep_threshold)] = 2
        classification[(raster > shallow_threshold) & (raster <= moderate_threshold)] = 1
        classification[np.isnan(raster)] = 0

        # Define classification mapping
        class_mapping = {
            1: 'shallow',
            2: 'moderate',
            3: 'deep'
        }

        # Extract polygons for each class
        geometries = []
        for class_value, class_name in class_mapping.items():
            mask = classification == class_value
            shapes_generator = shapes(mask.astype(np.int16), mask=mask, transform=transform)
            
            for geom, val in shapes_generator:
                geom_transformed = transform_geom(src.crs, 'EPSG:4326', geom)
                geometries.append({
                    'type': 'Feature',
                    'geometry': geom_transformed,
                    'properties': {
                        'classification': class_name
                    }
                })

    return geometries

def insert_flood_zones(db: Session, geometries: List[Dict]):
    try:
        batch_size = 1000  # can be adjusted based on need
        for i in range(0, len(geometries), batch_size):
            batch = geometries
            batch_data = [
                {
                    'classification': geom['properties']['classification'],
                    'geojson': geom['geometry']
                } for geom in batch
            ]
            db.bulk_insert_mappings(FloodZone, batch_data)
            db.commit()
    except Exception as e:
        db.rollback()
        logging.error(f"Error inserting flood zones: {e}")
        raise
