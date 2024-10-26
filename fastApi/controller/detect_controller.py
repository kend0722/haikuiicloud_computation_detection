import json
from typing import List

from fastapi import APIRouter, UploadFile, File, Form

from fastApi.router.custom_route import CustomRoute
from fastApi.schema.base_schema import RespListSchema, RespDetailSchema
from logger_conf import setup_logger
from yolo._hdfs import HDFSUtils
from yolo.helmet_detect import HelmetDetect

router = APIRouter(route_class=CustomRoute)

logger = setup_logger("detect_controller")


@router.post('/doDetect', response_model=RespDetailSchema)
async def do_detect_actually(file: UploadFile = File(...),
                             resizeW: int = Form(None), resizeH: int = Form(None),
                             poly: str = Form(None)) -> RespDetailSchema:
    resp = RespDetailSchema()
    try:
        img_bytes = await file.read()  # 读取上传的图片流
        helmet = HelmetDetect(poly=json.loads(poly) if poly else [], new_w=resizeW, new_h=resizeH)
        results = helmet.run(img_bytes)  # 执行推理
        # 返回推理结果
        resp.detail = results
        return resp
    except Exception as e:
        logger.error(e)
        resp.code = 500
        resp.message = str(e)
        return resp


@router.post('/doDetectFromHdfs', response_model=RespDetailSchema)
async def do_detect_from_hdfs(img: str = Form(...), resizeW: int = Form(None), resizeH: int = Form(None),
                              poly: str = Form(None)):
    resp = RespDetailSchema()
    try:
        img_bytes = HDFSUtils.download_bytes(img)  # 读取上传的图片流
        helmet = HelmetDetect(poly=json.loads(poly) if poly else [], new_w=resizeW, new_h=resizeH)
        results = helmet.run(img_bytes)  # 执行推理
        # 返回推理结果
        resp.detail = results
        return resp
    except Exception as e:
        logger.error(e)
        resp.code = 500
        resp.message = str(e)
        return resp
