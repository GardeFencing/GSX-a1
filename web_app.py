import atexit
import hashlib
import json
import logging
import os
import subprocess
import time
from functools import lru_cache
from uuid import uuid4

import redis
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

# Configure the root logger to DEBUG (or INFO if you want fewer logs)
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

gunicorn_logger = logging.getLogger("gunicorn.error")
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s"
)
for handler in gunicorn_logger.handlers:
    handler.setFormatter(formatter)

logger = app.logger

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max file size
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


try:
    redis_client = redis.StrictRedis(
        host="localhost", port=6379, db=0, decode_responses=True
    )
    redis_client.ping()
    redis_success = True
except redis.exceptions.ConnectionError:
    gunicorn_logger.exception("Failed to connect to redis")
    redis_success = False


def file_hash(video_path):
    hasher = hashlib.sha256()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_cached_results(cache_key: str) -> dict | None:
    if redis_success:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            return json.loads(cached_data)  # type: ignore
    return None


def cache_results(cache_key: str, data: tuple) -> None:
    if redis_success:
        redis_client.setex(cache_key, 3600, json.dumps(data))


def make_cache_key(args: list[str]):
    def parse(arg: str | list[str]) -> str:
        if isinstance(arg, list):
            arg = f"[{','.join(arg)}]"

        return arg

    return "_".join([parse(arg) for arg in args])


class VideoAnalyzer:
    def __init__(self, curr_video, conv_hist):
        # Paths to llama.cpp, model, etc.
        self.base_dir = os.path.abspath(os.getcwd())
        self.llama_cpp_path = os.path.expanduser("~/GSX-a1/llama.cpp")
        self.model_path = os.path.expanduser(
            "~/GSX-a1/llama.cpp/MiniCPM-V-2_6-gguf/ggml-model-Q4_K_M.gguf"
        )
        self.mmproj_path = os.path.expanduser(
            "~/GSX-a1/llama.cpp/MiniCPM-V-2_6-gguf/mmproj-model-f16.gguf"
        )
        self.books_dir = os.path.expanduser("~/GSX-a1/GardeRAG/knowledge_base/pdfs")

        # Keep track of current videos and chat histories
        self.current_video = curr_video
        self.conversation_history = conv_hist

        self._verify_critical_paths()
        self._load_fencing_knowledge()

        self.pipe_path = os.path.join(self.base_dir, "llama_input.pipe")
        self.process = None
        self.is_initialized = False

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

    @lru_cache(maxsize=128)
    def _run_inference(self, cmd: tuple) -> str:
        """
        Run the inference subprocess. We store `cmd` as a tuple so lru_cache
        can hash it. Alternatively, you could store results in a manual dict.
        """
        logger.info(f"Running command (cached): {' '.join(cmd)}")
        result = subprocess.run(
            cmd, cwd=self.llama_cpp_path, capture_output=True, text=False, check=True
        )

        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")

        logging.warning(f"STDERR: {stderr}")

        return stdout.strip()

    def analyze_video(self, video_path, query, raw_query):
        """Analyze the video using llama-minicpmv-cli with slightly reduced context size."""
        try:
            abs_video_path = os.path.abspath(video_path)
            if not os.path.exists(abs_video_path):
                raise RuntimeError(f"Video file not found at {abs_video_path}")

            logger.info(f"Analyzing video: {abs_video_path}")

            enhanced_prompt = self._enhance_prompt(query)

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

            logger.info(f"Full command: {' '.join(cmd)}")

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


def save_context():
    all_context = {
        "curr_vid": video_analyzer.current_video,
        "conv_hist": video_analyzer.conversation_history,
    }
    all_context = json.dumps(all_context)
    with open("save_context.json", "w") as file:
        file.write(all_context)


def rehydrate_state():
    try:
        with open("save_context.json", "r") as file:
            content = json.loads(file.read())
            current_video = content.get("curr_video", {})
            conversation_history = content.get("conv_hist", {})
    except (FileNotFoundError, json.JSONDecodeError):
        current_video, conversation_history = {}, {}

    return current_video, conversation_history


atexit.register(save_context)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    start_time = time.time()

    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    session_id = request.headers.get("X-Session-ID")
    if session_id is None or session_id == "null":
        session_id = str(uuid4())

    video_file = request.files["video"]
    if video_file.filename == "":
        logger.error("No file selected")
        return jsonify({"error": "No video file selected"}), 400

    try:
        # Log upload attempt
        logger.info(f"Processing video upload: {video_file.filename}")

        # Save the uploaded video
        filename = secure_filename(f"{session_id}-{video_file.filename}")
        video_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(video_path):
            video_file.save(video_path)

        default_query = """You are an expert fencing coach, describe what the fencer(s) is doing in simple, clear terms. 
        Focus on the actual movements first, then key observations about their form.
        Keep it conversational, like you're watching them in person.
        Format this as a structured coaching report."""

        hash_file = file_hash(video_path)
        cache_key = make_cache_key([hash_file, default_query])
        cached_results = get_cached_results(cache_key)
        if cached_results is not None:
            return jsonify(cached_results)

        # Log successful save
        logger.info(f"Video saved to: {video_path}")

        # Store the current video path
        video_analyzer.current_video[session_id] = video_path

        # Verify file exists
        if not os.path.exists(video_path):
            raise RuntimeError(f"Failed to save video file at {video_path}")

        # File size
        file_size = os.path.getsize(video_path)
        logger.info(f"Video file size: {file_size} bytes")

        # Initial analysis
        logger.info("Starting initial video analysis...")
        initial_analysis = video_analyzer.analyze_video(
            video_path,
            default_query,
        )
        logger.info(
            f"Initial analysis completed successfully. Length: {len(initial_analysis)}"
        )

        logger.info(f"Request took {time.time() - start_time}")

        response = {
            "success": True,
            "analysis": initial_analysis,
            "session_id": session_id,
        }

        cache_results(cache_key, response)

        return jsonify(response)

    except Exception as e:
        # Cleanup partial file if desired
        if os.path.exists(video_path):
            os.remove(video_path)
        logger.exception(f"Error processing video: {e}")

        logger.info(f"Request took {time.time() - start_time}")

        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    start_time = time.time()

    session_id = request.headers.get("X-Session-ID")
    if not video_analyzer.current_video.get(session_id):
        return jsonify({"error": "No video has been uploaded yet"}), 400

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    hash_file = file_hash(video_analyzer.current_video[session_id])
    cache_key = make_cache_key([hash_file, data["query"].lower().strip()])
    cached_results = get_cached_results(cache_key)
    if cached_results is not None:
        return jsonify(cached_results)

    try:
        logger.info(
            f"Processing chat query for video: {video_analyzer.current_video[session_id]}, query: {data['query']}"
        )

        # Incorporate conversation history
        conversation_history = video_analyzer.conversation_history.get(session_id, [])
        query_content = f"""
        You are an expert fencing coach having a conversation with your fencer.

        Conversation so far: {json.dumps(conversation_history)}

        The fencer says: {data['query']}
        """

        response = video_analyzer.analyze_video(
            video_analyzer.current_video[session_id],
            query_content,
            data["query"],
        )
        logger.info(f"Completed chat response. Length: {len(response)}")

        # Update conversation history
        if session_id not in video_analyzer.conversation_history:
            video_analyzer.conversation_history[session_id] = []

        video_analyzer.conversation_history[session_id].append(
            {
                "Fencer Question": data["query"],
                "Coach Answer": response,
            }
        )

        logger.info(f"Request took {time.time() - start_time}")

        response = {"response": response}

        cache_results(cache_key, response)

        return jsonify(response)

    except Exception as e:
        logger.exception(f"Error processing chat query: {e}")

        logger.info(f"Request took {time.time() - start_time}")

        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    curr_vid, conv_hist = rehydrate_state()
    video_analyzer = VideoAnalyzer(curr_vid, conv_hist)
    app.run(host="0.0.0.0", port=5000, debug=True)
