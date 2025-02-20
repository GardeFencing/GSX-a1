from gevent import monkey
monkey.patch_all()

import multiprocessing
from gunicorn.app.base import BaseApplication
from web_app import app
from config import Config

class GunicornApp(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        for key, value in self.options.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

def get_gunicorn_options():
    """Get Gunicorn server options optimized for EC2"""
    workers = Config.MAX_WORKERS
    threads = Config.MAX_THREADS
    
    return {
        'bind': f"{Config.HOST}:{Config.PORT}",
        'worker_class': 'gevent',
        'workers': workers,
        'threads': threads,
        'timeout': Config.REQUEST_TIMEOUT,
        'keepalive': 65,
        'max_requests': 1000,
        'max_requests_jitter': 50,
        'graceful_timeout': 30,
        'limit_request_line': 4096,
        'limit_request_fields': 100,
        'log_level': 'info',
        'capture_output': True,
        'enable_stdio_inheritance': True,
        'preload_app': True
    }

if __name__ == '__main__':
    # Initialize app with config
    Config.init_app(app)
    
    # Run with Gunicorn
    options = get_gunicorn_options()
    GunicornApp(app, options).run() 