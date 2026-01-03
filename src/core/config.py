"""Configuration management"""
from typing import List, Union

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore"
    )

    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Database - Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "ai-poweros-neo4j-2024"

    # Database - PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ai_poweros"
    POSTGRES_USER: str = "poweros"
    POSTGRES_PASSWORD: str = "ai-poweros-postgres-2024"

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: Union[str, List[str]] = "localhost:9092"
    KAFKA_TOPIC_EVENTS: str = "user-events"
    KAFKA_TOPIC_PREDICTIONS: str = "predictions"
    KAFKA_TOPIC_TASKS: str = "task-updates"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # InfluxDB
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: str = "ai-poweros-influx-token-2024"
    INFLUXDB_ORG: str = "ai-poweros"
    INFLUXDB_BUCKET: str = "metrics"

    # Security
    JWT_SECRET_KEY: str = "change-me"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_MINUTES: int = 30

    # Privacy
    DP_EPSILON: float = 1.0
    DP_DELTA: float = 1e-5

    # ML Models
    MODEL_CACHE_DIR: str = "./models"
    COREML_MODEL_PATH: str = "./models/routine_predictor.mlmodel"

    @field_validator("KAFKA_BOOTSTRAP_SERVERS", mode="before")
    @classmethod
    def parse_kafka_servers(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",")]
        return v


settings = Settings()
