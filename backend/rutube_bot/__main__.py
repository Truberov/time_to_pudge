from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rutube_bot.config import get_settings
from rutube_bot.utils.common import get_hostname
from rutube_bot.endpoints import list_of_routes


def bind_routes(application: FastAPI) -> None:
    """
    Bind all routes to application.
    """
    for route in list_of_routes:
        application.include_router(route)


def get_app() -> FastAPI:
    """
    Creates application and all dependable objects.
    """
    description = "RuTube QNA service"

    application = FastAPI(
        title="QNA service",
        description=description,
        docs_url="/docs",
        version="1.0.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    settings = get_settings()
    bind_routes(application)
    application.state.settings = settings
    return application


app = get_app()

if __name__ == "__main__":
    import uvicorn

    settings_for_application = get_settings()
    uvicorn.run(
        "rutube_bot.__main__:app",
        host=get_hostname(settings_for_application.APP_HOST),
        port=settings_for_application.APP_PORT,
        reload=True,
        log_level="debug",
    )
