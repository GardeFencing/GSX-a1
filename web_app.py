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

import sys
import io
import secrets
import re
import functools
from pathlib import Path

# Import GardeRAG components for knowledge enrichment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from GardeRAG.model import KeyframeSelector  # If available
    try:
        from GardeRAG.model import RAGModel  # Try to import RAGModel
        HAS_RAG_MODEL = True
        logger.info("Successfully imported RAGModel from GardeRAG")
    except (ImportError, AttributeError):
        HAS_RAG_MODEL = False
        logger.warning("RAGModel not available in GardeRAG")
except ImportError:
    logger.warning("GardeRAG components not available")
    HAS_RAG_MODEL = False

load_dotenv()

lock = Lock()
DOCKER: str | None = os.getenv("DOCKER")
MODEL: str | None = os.getenv("MODEL")
NODE_ENV: str | None = os.getenv("NODE_ENV")

# Configure the root logger to DEBUG (or INFO if you want fewer logs)
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, static_folder="templates", static_url_path="/templates")

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

# Access detailed model parameters
CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", "4096"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.8"))
TOP_K = int(os.environ.get("TOP_K", "100"))
REPEAT_PENALTY = float(os.environ.get("REPEAT_PENALTY", "1.05"))

# Flag to use LoRA fine-tuned model
USE_LORA = os.environ.get("USE_LORA", "false").lower() == "true"
LORA_MODEL_PATH = os.environ.get("LORA_MODEL_PATH", "MiniCPM-V/finetune/output/output_fencing_lora/merged_model")

def file_hash(file_obj):
    """Generate a unique hash for the file."""
    file_obj.seek(0)
    file_hash = hashlib.sha256(file_obj.read()).hexdigest()
    file_obj.seek(0)
    return file_hash


@functools.lru_cache(maxsize=128)
def get_cached_results(cache_key: str) -> dict | None:
    """Get cached results if available."""
    if redis_success:
        cached_data = redis_client.get(cache_key)
        return json.loads(cached_data) if cached_data else None
    return None


def cache_results(cache_key: str, data: tuple) -> None:
    """Cache results for future use."""
    if redis_success:
        redis_client.setex(cache_key, 86400, json.dumps(data))


def make_cache_key(args: list[str]):
    def parse(arg: str | list[str]) -> str:
        if isinstance(arg, list):
            return "_".join([parse(a) for a in arg])
        return str(arg).lower().strip()

    return "_".join([parse(arg) for arg in args])


class VideoAnalyzer:
    def __init__(self):
        # Paths to llama.cpp, model, etc.
        self.base_dir = os.path.abspath(os.getcwd())
        script_dir = os.path.dirname(os.path.abspath(__file__))

        self.llama_cpp_path = os.path.join(script_dir, "llama.cpp")
        
        # Select the appropriate model paths based on USE_LORA flag
        self.lora_enabled = False
        if USE_LORA and os.path.exists(os.path.join(script_dir, LORA_MODEL_PATH)):
            lora_dir = os.path.join(script_dir, LORA_MODEL_PATH)
            logger.info(f"Checking for LoRA fine-tuned model in {lora_dir}")
            
            # Look for model files with common extensions
            model_candidates = [
                "model.bin", 
                "pytorch_model.bin", 
                "model.safetensors", 
                "consolidated.00.pth",
                "pytorch_model.bin.index.json"
            ]
            
            model_found = False
            for candidate in model_candidates:
                candidate_path = os.path.join(lora_dir, candidate)
                if os.path.exists(candidate_path):
                    self.model_path = candidate_path
                    model_found = True
                    logger.info(f"Found LoRA model file: {candidate}")
                    break
            
            # Check for mmproj file
            mmproj_candidates = [
                "mmproj-model-f16.gguf",
                "mm_projector.bin",
                "mm_projector/model.safetensors"
            ]
            
            mmproj_found = False
            for candidate in mmproj_candidates:
                candidate_path = os.path.join(lora_dir, candidate)
                if os.path.exists(candidate_path):
                    self.mmproj_path = candidate_path
                    mmproj_found = True
                    logger.info(f"Found LoRA mmproj file: {candidate}")
                    break
            
            if model_found and mmproj_found:
                logger.info(f"Using LoRA fine-tuned model from {LORA_MODEL_PATH}")
                self.lora_enabled = True
            else:
                logger.warning(f"LoRA model files incomplete in {LORA_MODEL_PATH}, falling back to base model")
                # Fall back to the base model
                self._setup_base_model(script_dir)
        else:
            # Use base model
            self._setup_base_model(script_dir)
        
        # Path to fencing books for RAG
        self.books_dir = os.path.join(script_dir, "GardeRAG", "books")
        self.passages_path = os.path.join(script_dir, "GardeRAG", "fencing_passages.json")
        
        # Load RAG components if available
        self.rag_model = None
        self.has_rag = False
        
        if HAS_RAG_MODEL:
            try:
                self.rag_model = RAGModel()
                self.has_rag = True
                logger.info("RAG model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG model: {e}")
        
        self._verify_critical_paths()
        self._load_fencing_knowledge()

    def _setup_base_model(self, script_dir):
        """Set up paths for the base model."""
        logger.info("Using base MiniCPM-V model")
        self.model_path = os.path.join(
            script_dir, "models", "MiniCPM-V-2_6-gguf", MODEL
        )
        self.mmproj_path = os.path.join(
            script_dir, "models", "MiniCPM-V-2_6-gguf", "mmproj-model-f16.gguf"
        )

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
            
        # Check for video support
        minicpmv_cli_path = os.path.join(self.llama_cpp_path, "llama-minicpmv-cli")
        if not os.path.exists(minicpmv_cli_path):
            raise RuntimeError(f"llama-minicpmv-cli not found at {minicpmv_cli_path}")
        
        # Check if binary has video support by examining help output
        try:
            help_output = subprocess.check_output(
                [minicpmv_cli_path, "--help"],
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            if "--video" not in help_output:
                logger.warning("llama-minicpmv-cli may not have video support (--video flag not found in help output)")
        except subprocess.CalledProcessError:
            logger.warning("Failed to check llama-minicpmv-cli help output")

    def _load_fencing_knowledge(self):
        """Load key concepts from fencing books for RAG enhancement"""
        # Default context
        self.fencing_context = """Key fencing concepts to consider:
        - Proper stance and footwork are fundamental
        - Balance and control in movement
        - Blade control and point control
        - Distance management and timing
        - Technical precision in execution"""

        # Load specific fencing passages if available
        self.fencing_passages = []
        try:
            if os.path.exists(self.passages_path):
                logger.info(f"Loading pre-processed fencing passages from {self.passages_path}")
                with open(self.passages_path, 'r') as f:
                    self.fencing_passages = json.load(f)
                logger.info(f"Loaded {len(self.fencing_passages)} pre-processed fencing passages")
            elif os.path.exists(self.books_dir):
                logger.info(f"Found fencing books directory at {self.books_dir}")
                logger.warning(f"No pre-processed passages found at {self.passages_path}, consider running extract_fencing_knowledge.py")
        except Exception as e:
            logger.warning(f"Could not load fencing passages: {e}")

    def _enhance_prompt_with_rag(self, video_path: str, query: str) -> str:
        """Enhance the prompt with relevant fencing knowledge using RAG"""
        # Base system context
        system_context = f"""You are an expert fencing coach with deep knowledge of classical and modern fencing techniques.
        {self.fencing_context}
        
        When analyzing video, focus on:
        1. Technical execution and form
        2. Movement efficiency and control
        3. Areas for improvement
        
        Keep your analysis clear and specific."""
        
        # If RAG is available, use it to enhance the prompt
        if self.has_rag and self.rag_model:
            try:
                # Extract key frames from video for content-based retrieval
                relevant_passages = self.rag_model.retrieve_relevant_passages(
                    query=query,
                    top_k=3
                )
                
                if relevant_passages:
                    rag_context = "\n\nRelevant fencing knowledge:\n"
                    for i, passage in enumerate(relevant_passages):
                        rag_context += f"[{i+1}] {passage['text']}\n"
                    
                    system_context += rag_context
                    logger.info(f"Enhanced prompt with {len(relevant_passages)} RAG passages")
            except Exception as e:
                logger.warning(f"Error enhancing prompt with RAG: {e}")
        # If RAG is not available but we have fencing passages, use basic retrieval
        elif self.fencing_passages:
            # Simple keyword matching for relevant passages
            query_terms = set(query.lower().split())
            
            # Find passages that match the query terms
            matched_passages = []
            for passage in self.fencing_passages:
                passage_text = passage.lower() if isinstance(passage, str) else passage.get('text', '').lower()
                passage_terms = set(passage_text.split())
                # Calculate simple overlap score
                overlap = len(query_terms.intersection(passage_terms))
                if overlap > 0:
                    matched_passages.append((passage, overlap))
            
            # Sort by relevance and take top 3
            matched_passages.sort(key=lambda x: x[1], reverse=True)
            top_passages = [p[0] for p in matched_passages[:3]]
            
            if top_passages:
                rag_context = "\n\nRelevant fencing knowledge:\n"
                for i, passage in enumerate(top_passages):
                    if isinstance(passage, str):
                        rag_context += f"[{i+1}] {passage}\n"
                    else:
                        rag_context += f"[{i+1}] {passage.get('text', '')}\n"
                
                system_context += rag_context
                logger.info(f"Enhanced prompt with {len(top_passages)} matched passages")

        if self.lora_enabled:
            # If using LoRA model, add a hint that it was fine-tuned for fencing
            system_context += "\n\nYou are using a model specifically fine-tuned on fencing techniques. Provide detailed analysis of fencing actions."

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
        """Analyze the video using llama-minicpmv-cli with RAG enhancement."""
        try:
            abs_video_path = os.path.abspath(video_path)
            if not os.path.exists(abs_video_path):
                raise RuntimeError(f"Video file not found at {abs_video_path}")

            logger.info(f"Analyzing video: {abs_video_path}")

            if enhance:
                # Use RAG-enhanced prompt
                enhanced_prompt = self._enhance_prompt_with_rag(abs_video_path, query)
            else:
                enhanced_prompt = query

            # Configure model parameters
            cmd = (
                "./llama-minicpmv-cli",
                "-m",
                self.model_path,
                "--mmproj",
                self.mmproj_path,
                "-c",
                str(CONTEXT_SIZE),
                "--temp",
                str(TEMPERATURE),
                "--top-p",
                str(TOP_P),
                "--top-k",
                str(TOP_K),
                "--repeat-penalty",
                str(REPEAT_PENALTY),
                "--video",
                abs_video_path,
                "-p",
                enhanced_prompt,
            )

            # Run or Retrieve from Cache
            cache_key = make_cache_key([
                "video",
                os.path.basename(abs_video_path), 
                enhanced_prompt,
                "lora" if self.lora_enabled else "base"
            ])
            
            cached_results = get_cached_results(cache_key)
            if cached_results is not None:
                logger.info("Using cached video analysis results")
                return cached_results
                
            output = self._run_inference(cmd)

            if not output:
                raise RuntimeError("Failed to analyze video. Output is empty.")

            logger.info(f"Command completed successfully, output length: {len(output)}")
            
            # Cache the result
            cache_results(cache_key, output)
            
            return output

        except subprocess.CalledProcessError as e:
            logger.exception(f"Error running video analysis: {e.stderr}")
            
            # If the error might be related to video support with LoRA model
            error_str = e.stderr.decode("utf-8", errors="replace") if hasattr(e, "stderr") else str(e)
            if self.lora_enabled and ("video" in error_str.lower() or "format" in error_str.lower()):
                logger.warning("Error may be related to video compatibility with LoRA model. Falling back to base model.")
                # Temporarily switch to base model and try again
                old_model_path = self.model_path
                old_mmproj_path = self.mmproj_path
                self._setup_base_model(os.path.dirname(os.path.abspath(__file__)))
                try:
                    logger.info("Retrying with base model...")
                    result = self.analyze_video(video_path, query, enhance)
                    logger.info("Analysis succeeded with base model")
                    return result
                except Exception as retry_error:
                    logger.exception(f"Retry also failed: {retry_error}")
                    raise RuntimeError(f"Video analysis failed with both LoRA and base models")
                finally:
                    # Restore paths
                    self.model_path = old_model_path
                    self.mmproj_path = old_mmproj_path
            
            raise RuntimeError(f"Video analysis failed: {error_str}")
            
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

        logger.info(f"Video saved as: {hashed_filename}")
        return hashed_filename
    except requests.exceptions.RequestException as e:
        logger.exception(f"Failed to fetch video: {e}")
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

        logger.info(f"Response {response}")

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
    logger.info(f"HASHED FILE {hash_file}")

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

        logger.info(f"Response {response}")

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
