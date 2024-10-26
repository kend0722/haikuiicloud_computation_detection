from fastapi import APIRouter

from fastApi.controller.detect_controller import router

api_router = APIRouter()
api_router.include_router(router)