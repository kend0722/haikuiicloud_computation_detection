import sys
import os

import uvicorn

from fastApi.main import app

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)


