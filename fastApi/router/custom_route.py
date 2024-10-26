from fastapi.routing import APIRoute
from typing import Callable
from starlette.requests import Request
from starlette.responses import Response

class CustomRoute(APIRoute):
    """
    自定义APIRouter
    """

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            # 记录日志
            response = await original_route_handler(request)
            return response

        return custom_route_handler