import datetime

from fastapi import APIRouter
from fastApi.config.FastapiConfig import FastapiConfig

base_router = APIRouter()


@base_router.get('/')
async def get_root():
    """
    访问根路径
    """
    return {
        'title': FastapiConfig.title,
        'description': FastapiConfig.description,
        'version': FastapiConfig.version,
        'time': datetime.datetime.now()
    }
