from typing import List, Dict

from pydantic import BaseModel

from fastApi.utils.json_encoders import JSONEncoders


class BaseSchema(BaseModel):
    """
    基础Schema
    """

    class Config:
        json_encoders = JSONEncoders.json_encoders  # 使用自定义json转换


class RespBaseSchema(BaseSchema):
    """
    基础返回Schema
    """
    code: int = 0  # 返回编号
    message: str = 'SUCCESS'  # 返回消息


class RespDetailSchema(RespBaseSchema):
    """
    详情返回Schema
    """
    detail: dict = None  # 返回详情

class RespListSchema(RespBaseSchema):
    """
    列表返回Schema
    """
    list: List[Dict] = None  # 数据list

class RespListPageSchema(RespBaseSchema):
    """
    列表返回Schema
    """
    page: int = 0  # 当前页码
    size: int = 0  # 每页大小
    count: int = 0  # 数据总条数
    page_count: int = 0  # 总页数
    list: List[Dict] = None  # 数据list
