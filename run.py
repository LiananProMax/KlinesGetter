#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

# 将项目根目录添加到Python路径以允许从'app'导入
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.main_app import main_application

if __name__ == "__main__":
    main_application()
