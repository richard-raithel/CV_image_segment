import os


ALLOWED_HOSTS = [os.environ['WEBSITE_HOSTNAME']] if 'WEBSITE_HOSTNAME' in os.environ else []

DATABASES = {
    'default': {
        'ENGINE': 'djongo',
        'NAME': 'computervisionimages-results',
        'ENFORCE_SCHEMA': False,
        'CLIENT': {
            'host': os.getenv("COSMOS_URI")
        }
    },
}

SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True