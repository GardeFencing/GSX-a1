from gevent import monkey

monkey.patch_all()

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from config import Config
from web_app import app

if __name__ == "__main__":
    # Initialize app with config
    Config.init_app(app)

    # Run with Flask's development server if not running with Gunicorn
    app.run(host="0.0.0.0", port=5000)
