from .health_check import api_router as health_check_router

list_of_routes = [
    health_check_router,
]

list_of_langserve_routes = []

__all__ = [
    "list_of_routes",
    "list_of_langserve_routes",
]
