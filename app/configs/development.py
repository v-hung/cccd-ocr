from .base import *
import os

DEBUG = True
DATABASE_URL = os.getenv('DEV_DATABASE_URL', 'sqlite:///dev.db')
SECRET_KEY = 'dev-secret-key'

GEMINI_KEY = os.getenv('DEV_GEMINI_KEY', '')