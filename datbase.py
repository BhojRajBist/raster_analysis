from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,Session
from typing import Generator
from models import FloodZoneH3

SQLALCHEMY_DATABASE_URL = 'postgresql://postgres:postgres@localhost/practice'

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    new_flood_zone = FloodZoneH3(h3_index='8928308280fffff', classification='shallow')
    db.add(new_flood_zone)
    db.commit()
    try:
        yield db
    finally:
        db.close()