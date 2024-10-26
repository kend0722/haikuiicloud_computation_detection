import json
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from fastApi.config.FastapiConfig import FastapiConfig
from fastapi.exceptions import RequestValidationError
from starlette.responses import Response
from starlette.middleware.cors import CORSMiddleware

from fastApi.execeptions.business_exeception import BusinessException
from fastApi.router.api_router import api_router
from fastApi.controller.base_controller import base_router
from yolo.model.model_rep import ModelRep, PersonModel, HelmetModel


app = FastAPI(**FastapiConfig.__dict__)
@app.on_event("startup")
async def startup():
    if not ModelRep.contains_model(PersonModel.model_key):
        person_model = PersonModel()
        person_model.init_model()
        ModelRep.register_model(PersonModel.model_key, person_model)
    if not ModelRep.contains_model(HelmetModel.model_key):
        helmet_model = HelmetModel()
        helmet_model.init_model()
        ModelRep.register_model(HelmetModel.model_key, helmet_model)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return Response(json.dumps({
        'code': 400,
        'message': str(exc),
    }), status_code=400)

@app.exception_handler(Exception)
async def http_exception_handler(request, exc):
    return Response(json.dumps({
        'code': 500,
        'message': str(exc),
    }), status_code=500)

@app.exception_handler(BusinessException)
async def http_exception_handler(request, exc):
    return Response(json.dumps({
        'code': exc.error_code,
        'message': str(exc),
    }), status_code=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(base_router)
app.include_router(api_router)
app.mount('/static', StaticFiles(directory=FastapiConfig.res_path), name='static')
