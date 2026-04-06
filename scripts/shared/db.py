"""
TimescaleDB connection helper.
Loads credentials from .env and returns SQLAlchemy engine / psycopg2 connection.
"""
import os
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine

load_dotenv()

def get_connection_string() -> str:
    host = os.environ["TIMESCALE_HOST"]
    port = os.environ.get("TIMESCALE_PORT", "5432")
    db = os.environ["TIMESCALE_DB"]
    user = os.environ["TIMESCALE_USER"]
    password = os.environ["TIMESCALE_PASSWORD"]
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def get_engine():
    return create_engine(get_connection_string())

def get_psycopg2_conn():
    return psycopg2.connect(get_connection_string())
