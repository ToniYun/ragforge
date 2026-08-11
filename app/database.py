from sqlalchemy import URL, create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from app.config import settings

url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password=settings.postgres_password,
    host="localhost",
    port=5432,
    database="ragforge"
)

engine = create_engine(url)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()