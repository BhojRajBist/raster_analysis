from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from datbase import SessionLocal
from classify_flood_extent import classify_flood_extent, insert_flood_zones
from h3_indexing import convert_to_h3, insert_h3_data
from models import FloodZoneH3

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/process_raster/")
async def process_raster(file_path: str, db: Session = Depends(get_db)):
    # Step 1: Classify the raster
    geometries = classify_flood_extent(file_path)
    new_flood_zone = FloodZoneH3(h3_index='8928308280fffff', classification='shallow')
    db.add(new_flood_zone)
    db.commit()
    
    # Step 2: Insert classified flood zones into the database
    insert_flood_zones(db, geometries)
    
    # Step 3: Convert to H3
    h3_geometries = convert_to_h3(geometries, resolution=8)
    
    # Step 4: Insert H3 data into the database
    insert_h3_data(db, h3_geometries)
    
    return {"status": "Processing completed"}
