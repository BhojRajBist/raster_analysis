from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
import geoalchemy2.types

Base = declarative_base()

class FloodZone(Base):
    __tablename__ = 'flood_zones'
    
    id = Column(Integer, primary_key=True, index=True)
    classification = Column(String, index=True)
    geojson = Column(geoalchemy2.types.Geometry(geometry_type='GEOMETRY', srid=4326))


# CREATE TABLE flood_zones (
#     id SERIAL PRIMARY KEY,
#     classification VARCHAR(255) NOT NULL,
#     geojson GEOMETRY(GEOMETRY, 4326) -- SRID 4326 for WGS 84
# );

class FloodZoneH3(Base):
    __tablename__ = 'flood_zones_h3'
    
    id = Column(Integer, primary_key=True, index=True)
    h3_index = Column(String, index=True)
    classification = Column(String)

# CREATE TABLE flood_zones_h3 (
#     id SERIAL PRIMARY KEY,
#     h3_index VARCHAR NOT NULL,
#     classification VARCHAR NOT NULL
# );

