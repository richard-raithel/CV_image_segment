from .base import *

# set "DJANGO_CONFIG = 'prod'" as an environment variable on production machine
# set "DJANGO_CONFIG = 'dev'" as an environment variable on development machine

# os.environ['DJANGO_CONFIG'] = 'prod'
# print(Path(__file__).resolve().parent.parent.parent)

# if os.environ['DJANGO_CONFIG'] == 'prod':
#     from .prod import *
# else:
#     from .dev import *

if 'DJANGO_CONFIG' in os.environ:
    if os.environ['DJANGO_CONFIG'] == 'dev':
        from .dev import *
    else:
        from .prod import *
else:
    from .prod import *
