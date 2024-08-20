from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy import create_engine, MetaData, Table, select, distinct
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from databases import Database

DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost/practice"

database = Database(DATABASE_URL)
metadata = MetaData()

# Define the wards table
wards = Table(
    "wards", metadata,
    autoload_with=create_engine(DATABASE_URL.replace("asyncpg", "psycopg2"))
)

engine = create_async_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

app = FastAPI()

# Dependency
async def get_db():
    async with SessionLocal() as session:
        yield session
