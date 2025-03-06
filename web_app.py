import hashlib
import json
import logging
import os
import subprocess
import time
import uuid
from multiprocessing import Lock

import redis
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

load_dotenv()

lock = Lock()
DOCKER: str | None = os.getenv("DOCKER")
MODEL: str | None = os.getenv("MODEL")
NODE_ENV: str | None = os.getenv("NODE_ENV")

# Configure the root logger to DEBUG (or INFO if you want fewer logs)
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

if NODE_ENV == "dev":
    CORS(
        app,
        origins=["http://localhost:3000"],
        supports_credentials=True,
        methods=["POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

if "gunicorn" in os.environ.get("SERVER_SOFTWARE", ""):
    # Running under Gunicorn
    app.logger = logging.getLogger("gunicorn.error")
    app.logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
    )
    for handler in app.logger.handlers:
        handler.setFormatter(formatter)
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    )
    app.logger = logging.getLogger(__name__)

logger = app.logger
logger.info("Logging setup complete!")

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max file size
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

redis_host = "redis" if DOCKER == "true" else "localhost"
try:
    redis_client = redis.StrictRedis(
        host=redis_host, port=6379, db=0, decode_responses=True
    )
    redis_client.ping()
    redis_success = True
except redis.exceptions.ConnectionError:
    logger.exception("Failed to connect to redis")
    redis_success = False


def file_hash(file_obj):
    hasher = hashlib.sha256()
    for chunk in iter(lambda: file_obj.read(4096), b""):
        hasher.update(chunk)
    file_obj.seek(0)
    return hasher.hexdigest()


def get_cached_results(cache_key: str) -> dict | None:
    if redis_success:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)  # type: ignore
    return None


def cache_results(cache_key: str, data: tuple) -> None:
    if redis_success:
        redis_client.setex(cache_key, 86400, json.dumps(data))


def make_cache_key(args: list[str]):
    def parse(arg: str | list[str]) -> str:
        if isinstance(arg, list):
            arg = f"[{','.join(arg)}]"

        return arg

    return "_".join([parse(arg) for arg in args])


class VideoAnalyzer:
    def __init__(self):
        # Paths to llama.cpp, model, etc.
        self.base_dir = os.path.abspath(os.getcwd())
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.llama_cpp_path = os.path.join(script_dir, "llama.cpp")
        self.model_path = os.path.join(
            script_dir, "models", "MiniCPM-V-2_6-gguf", MODEL
        )
        self.mmproj_path = os.path.join(
            script_dir, "models", "MiniCPM-V-2_6-gguf", "mmproj-model-f16.gguf"
        )
        self.books_dir = os.path.join(script_dir, "GardeRAG", "knowledge_base", "pdfs")

        self._verify_critical_paths()
        self._load_fencing_knowledge()

    def _verify_critical_paths(self):
        """Verify that critical paths exist (models and llama.cpp)"""
        if not os.path.exists(self.llama_cpp_path):
            raise RuntimeError(
                f"llama.cpp directory not found at {self.llama_cpp_path}"
            )
        if not os.path.exists(self.model_path):
            raise RuntimeError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.mmproj_path):
            raise RuntimeError(f"MM project model not found at {self.mmproj_path}")

    def _load_fencing_knowledge(self):
        """Load key concepts from fencing books or use default if not available"""
        # Default context
        self.fencing_context = """Key fencing concepts to consider:
        - Proper stance and footwork are fundamental
        - Balance and control in movement
        - Blade control and point control
        - Distance management and timing
        - Technical precision in execution"""

        # (Optional) check for books and log a message
        try:
            if os.path.exists(self.books_dir):
                logger.info(
                    f"Found books directory at {self.books_dir}, but using default context for now"
                )
                # Future enhancement: Load and process books when available
        except Exception as e:
            logger.warning(
                f"Could not access books directory: {e}, using default context"
            )

    def _enhance_prompt(self, query: str) -> str:
        """Enhance the prompt with fencing context"""
        system_context = f"""You are an expert fencing coach with deep knowledge of classical and modern fencing techniques.
        {self.fencing_context}
        
        When analyzing video, focus on:
        1. Technical execution and form
        2. Movement efficiency and control
        3. Areas for improvement
        
        Keep your analysis clear and specific."""

        return f"{system_context}\n\nAnalyze this video: {query.lower().strip()}"

    def _run_inference(self, cmd: tuple) -> str:
        """
        Run the inference subprocess. We store `cmd` as a tuple so lru_cache
        can hash it. Alternatively, you could store results in a manual dict.
        """
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=self.llama_cpp_path, capture_output=True, text=False, check=True
        )

        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")

        logging.warning(f"STDERR: {stderr}")

        return stdout.strip()

    def analyze_video(self, video_path, query, enhance=True):
        """Analyze the video using llama-minicpmv-cli with slightly reduced context size."""
        try:
            abs_video_path = os.path.abspath(video_path)
            if not os.path.exists(abs_video_path):
                raise RuntimeError(f"Video file not found at {abs_video_path}")

            logger.info(f"Analyzing video: {abs_video_path}")

            if enhance:
                enhanced_prompt = self._enhance_prompt(query)
            else:
                enhanced_prompt = query

            # Do not lower context length (-c), will bug out
            cmd = (
                "./llama-minicpmv-cli",
                "-m",
                self.model_path,
                "--mmproj",
                self.mmproj_path,
                "-c",
                "4096",
                "--temp",
                "0.7",
                "--top-p",
                "0.8",
                "--top-k",
                "100",
                "--repeat-penalty",
                "1.05",
                "--video",
                abs_video_path,
                "-p",
                enhanced_prompt,
            )

            # 9) Run or Retrieve from Cache
            output = self._run_inference(cmd)

            if not output:
                raise RuntimeError("Failed to analyze video. Output is empty.")

            logger.info(f"Command completed successfully, output length: {len(output)}")
            return output

        except subprocess.CalledProcessError as e:
            logger.exception(f"Error running video analysis: {e.stderr}")
            raise RuntimeError(f"Video analysis failed: {e.stderr}")
        except Exception as e:
            logger.exception(f"Unexpected error during video analysis: {e}")
            raise


video_analyzer = VideoAnalyzer()


@app.route("/analyze")
def index():
    return render_template("index.html")


@app.route("/analyze/health")
def health():
    return jsonify({"success": True})


def fetch_video(video_url, save_dir):
    temp_file = os.path.join(save_dir, f"{uuid.uuid4()}")

    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

        # Step 1: Download video and save as a temp file
        with open(temp_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Step 2: Compute SHA-256 hash of the downloaded file
        hasher = hashlib.sha256()
        with open(temp_file, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                hasher.update(chunk)
        video_hash = hasher.hexdigest()

        # Step 3: Rename the file with its hash
        hashed_filename = os.path.join(save_dir, f"{video_hash}")
        if not os.path.exists(hashed_filename):
            os.rename(temp_file, hashed_filename)

        print(f"Video saved as: {hashed_filename}")
        return hashed_filename
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch video: {e}")
        return None


@app.route("/analyze/upload", methods=["POST"])
def upload_video():
    start_time = time.time()

    try:
        video_file = request.files["video"]
        if video_file.filename == "":
            logger.error("No file selected")
            return jsonify({"error": "No video file selected", "success": False}), 400

        hashed = file_hash(video_file)
        hash_file = os.path.join(app.config["UPLOAD_FOLDER"], hashed)
        if not os.path.exists(hash_file):
            video_file.save(hash_file)

        # data = request.get_json()
        # video_url = data.get("videoUrl")

        # hash_file = fetch_video(video_url, app.config["UPLOAD_FOLDER"])

        # Log upload attempt
        logger.info("Processing video upload")

        default_query = """You are an expert fencing coach, describe what the fencer(s) is doing in simple, clear terms. 
        Focus on the actual movements first, then key observations about their form.
        Keep it conversational, like you're watching them in person.
        Format this as a structured coaching report."""

        # Log successful save
        logger.info(f"Video saved to: {hash_file}")

        # Verify file exists
        if not os.path.exists(hash_file):
            raise RuntimeError(f"Failed to save video file at {hash_file}")

        # File size
        file_size = os.path.getsize(hash_file)
        logger.info(f"Video file size: {file_size} bytes")

        cache_key = make_cache_key([hash_file.split("/")[1], default_query])
        cached_results = get_cached_results(cache_key)
        if cached_results is not None:
            logger.info(f"Using cached results, {cached_results}")

            logger.info(f"Request took {time.time() - start_time}")

            if os.path.exists(hash_file):
                os.remove(hash_file)

            return jsonify(cached_results)

        # Initial analysis
        logger.info("Starting initial video analysis...")
        initial_analysis = video_analyzer.analyze_video(
            hash_file,
            default_query,
        )
        logger.info(
            f"Initial analysis completed successfully. Length: {len(initial_analysis)}"
        )

        response = {
            "success": True,
            "analysis": initial_analysis,
        }

        cache_results(cache_key, response)

        logger.info(f"Request took {time.time() - start_time}")

        if os.path.exists(hash_file):
            os.remove(hash_file)

        return jsonify(response)

    except Exception as e:
        # Cleanup partial file if desired
        logger.exception(f"Error processing video: {e}")

        logger.info(f"Request took {time.time() - start_time}")

        if os.path.exists(hash_file):
            os.remove(hash_file)

        return jsonify({"error": str(e), "success": False}), 500


def format_conversation_history(conversation_history):
    formatted_history = ""
    for exchange in conversation_history:
        formatted_history += f"{exchange.get('sender')}: {exchange.get('message')}\n"
    return formatted_history.strip()


@app.route("/analyze/chat", methods=["POST"])
def chat():
    start_time = time.time()

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided", "success": False}), 400

    video_url = data.get("videoUrl")
    query = data.get("query")
    previous_history = data.get("messages", [])

    hash_file = fetch_video(video_url, app.config["UPLOAD_FOLDER"])
    print("HASHED FILE", hash_file)

    if not os.path.exists(hash_file):
        return (
            jsonify({"error": "No video has been uploaded yet", "success": False}),
            400,
        )

    cache_key = make_cache_key([hash_file.split("/")[1], data["query"].lower().strip()])
    cached_results = get_cached_results(cache_key)
    if cached_results is not None:
        logger.info(f"Using cached results, {cached_results}")

        logger.info(f"Request took {time.time() - start_time}")

        if os.path.exists(hash_file):
            os.remove(hash_file)

        return jsonify(cached_results)

    try:
        logger.info(f"Processing chat query for video: {hash_file}, query: {query}")

        query_content = f"""
        You are an expert fencing coach having a conversation with your fencer.

        Conversation so far: 
        {format_conversation_history(previous_history)}

        The fencer says: {query}
        """

        response = video_analyzer.analyze_video(
            hash_file,
            query_content,
            enhance=False,
        )
        logger.info(f"Completed chat response. Length: {len(response)}")

        response = {"analysis": response, "success": True}

        cache_results(cache_key, response)

        logger.info(f"Request took {time.time() - start_time}")

        if os.path.exists(hash_file):
            os.remove(hash_file)

        return jsonify(response)

    except Exception as e:
        logger.exception(f"Error processing chat query: {e}")

        logger.info(f"Request took {time.time() - start_time}")

        if os.path.exists(hash_file):
            os.remove(hash_file)

        return jsonify({"error": str(e), "success": False}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
