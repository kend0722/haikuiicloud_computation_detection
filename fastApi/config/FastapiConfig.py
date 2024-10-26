# 配置文件
class FastapiConfig:
    debug = True
    title = 'FastAPI Yolo'
    description = '模型检测API项目'
    version = '0.0.1'
    openapi_url = '/openapi.json'
    openapi_prefix = ''
    docs_url = '/docs'
    redoc_url = '/redoc'
    swagger_ui_oauth2_redirect_url = '/docs/oauth2-redirect'
    swagger_ui_init_oauth = None
    res_path = 'static'
