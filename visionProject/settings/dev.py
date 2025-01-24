import os
from dotenv import load_dotenv
from os.path import join, dirname

load_dotenv()


ALLOWED_HOSTS = ["localhost", "127.0.0.1"]


DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'computervisionimages-results',
        'ENFORCE_SCHEMA': False,
        'CLIENT': {
            'host': os.getenv("MONGO_URI")
        }
    }
}


SECURE_SSL_REDIRECT = False
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
