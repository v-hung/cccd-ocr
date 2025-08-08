from .base import *
import os

DEBUG = False
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///dev.db')
SECRET_KEY = 'dev-secret-key'