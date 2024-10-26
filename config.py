import os

import yaml

env = os.getenv('APP_ENV', 'dev')  # 默认使用开发环境
config_file = f'config_{env}.yaml'
with open(config_file, "r") as f:
    config = yaml.safe_load(f)