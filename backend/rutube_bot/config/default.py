from os import environ


class DefaultSettings:
    PATH_PREFIX: str = environ.get("PATH_PREFIX", "/api/v1")
    APP_HOST: str = environ.get("APP_HOST", "http://0.0.0.0")
    APP_PORT: int = int(environ.get("APP_PORT", 8080))
    OLLAMA_URL: str = environ.get("OLLAMA_URL", "http://192.144.12.76:11434")
    REDIS_URL: str = environ.get("REDIS_URL", "redis://0.0.0.0:6379")
