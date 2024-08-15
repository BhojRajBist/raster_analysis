# import h3
# import pyarrow as pa
# from h3ronpy.arrow.raster import raster_to_dataframe, nearest_h3_resolution
# from sqlalchemy.orm import Session
# from models import FloodZoneH3
# import logging
# from typing import List, Dict


# def convert_to_h3(geometries: List[Dict], resolution: int) -> List[Dict]:
#     """Convert GeoJSON geometries to H3 indices."""
#     h3_geometries = []
#     for feature in geometries:
#         geometry = feature['geometry']
#         classification = feature['properties']['classification']
        
#         # Convert to H3 index
#         h3_index = h3.geo_to_h3(geometry['coordinates'][1], geometry['coordinates'][0], resolution)
        
#         h3_geometries.append({
#             'h3_index': h3_index,
#             'classification': classification
#         })
#     return h3_geometries

# def insert_h3_data(db: Session, h3_geometries: List[Dict]):
#     try:
#         batch_size = 1000  # can be adjusted based on need
#         for i in range(0, len(h3_geometries), batch_size):
#             batch = h3_geometries[i:i + batch_size]
#             batch_data = [
#                 {
#                     'h3_index': geom['h3_index'],
#                     'classification': geom['classification']
#                 } for geom in batch
#             ]
#             db.bulk_insert_mappings(FloodZoneH3, batch_data)
#             db.commit()
#     except Exception as e:
#         db.rollback()
#         logging.error(f"Error inserting H3 data: {e}")
#         raise
