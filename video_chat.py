import os
import cv2
import torch
import numpy as np
import faiss
from openai import OpenAI
import logging
from PIL import Image
import requests
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------------
# Load SentenceTransformer for caption embeddings
# ------------------------------------------------------------------------------
logger.info("Loading sentence transformer for caption embeddings...")
sent_transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device)

def encode_image_base64(pil_image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def get_rag_caption(image, query="Describe this frame in detail, focusing on any fencing techniques, stances, or movements visible."):
    """
    Get caption from custom RAG endpoint
    """
    try:
        # Convert PIL image to base64
        img_base64 = encode_image_base64(image)
        
        # Prepare the request
        url = "http://54.227.20.42/query"
        payload = {
            "query": query,
            "image": img_base64
        }
        
        # Make the request
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Extract caption from response
        result = response.json()
        return result.get("response", "No caption generated")
    except Exception as e:
        logger.error(f"Error getting RAG caption: {e}")
        return "Error generating caption"

def extract_frames(video_path, frame_interval=3):
    """
    Extract frames from the video more efficiently by using cv2 seek.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError("FPS is 0; cannot compute frame interval.")
        
    # Calculate frame positions to extract
    interval = int(fps * frame_interval)
    frame_positions = range(0, total_frames, interval)
    frames = []
    
    for pos in frame_positions[:10]:  # Limit to 10 frames for speed
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize image to reduce memory usage and processing time
            height, width = frame_rgb.shape[:2]
            new_height = 224
            new_width = int(width * (new_height / height))
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
    
    cap.release()
    logger.info(f"Extracted {len(frames)} frames from the video.")
    return frames

def generate_captions(frames, focus_topic=None):
    """
    Generate captions using custom RAG endpoint
    """
    captions = []
    logger.info("Generating captions using custom RAG...")
    
    # Different queries to get varied descriptions
    queries = [
        "What fencing stance or position is shown in this frame?",
        "What fencing equipment or gear is visible in this frame?",
        "What fencing technique or movement is being demonstrated?",
        "Describe the fencer's form and posture in this frame.",
        "What stage of a fencing move is shown in this frame?"
    ]
    
    for idx, img in enumerate(frames):
        # Get two different perspectives for each frame
        selected_queries = np.random.choice(queries, size=2, replace=False)
        frame_captions = []
        
        for query in selected_queries:
            caption = get_rag_caption(img, query)
            if caption and caption not in frame_captions:
                frame_captions.append(caption)
        
        # Combine the captions
        final_caption = " ".join(frame_captions) if frame_captions else "No caption generated"
        captions.append(final_caption)
        logger.info(f"Frame {idx+1} caption: {final_caption}")
    
    return captions

def build_caption_index(captions):
    """
    Embed captions with a SentenceTransformer and build a FAISS index.
    """
    logger.info("Embedding captions and building FAISS index...")
    caption_embeddings = sent_transformer.encode(captions, convert_to_numpy=True, normalize_embeddings=True)
    d = caption_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Using inner product (cosine similarity on normalized vectors)
    index.add(caption_embeddings)
    return index, caption_embeddings

def query_captions(query, index, captions, top_k=3):
    """
    Embed the query and retrieve the top_k matching captions.
    Returns a list of (caption, similarity score) tuples.
    """
    query_embedding = sent_transformer.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, idxs = index.search(query_embedding, top_k)
    results = []
    for j, i in enumerate(idxs[0]):
        results.append((captions[i], scores[0][j]))
    return results

class VideoChat:
    def __init__(self, video_path: str, openai_api_key: str, frame_interval=3, focus_topic="fencing"):
        """
        Initialize the VideoChat system with RAG-based captioning and semantic search.
        
        Parameters:
            video_path (str): Path to the video file.
            openai_api_key (str): Your OpenAI API key.
            frame_interval (int): Seconds between sampled frames.
            focus_topic (str): Topic to focus on in the captions (e.g., "fencing").
        """
        # Initialize OpenAI client
        self.client = OpenAI(api_key=openai_api_key)
        
        # Extract frames using the specified interval
        self.frames = extract_frames(video_path, frame_interval=frame_interval)
        if not self.frames:
            raise ValueError("No frames extracted from the video.")
        
        # Generate captions using RAG
        self.captions = generate_captions(self.frames, focus_topic)
        
        # Build semantic search index over the generated captions
        self.index, _ = build_caption_index(self.captions)
        
        # System prompt for GPT-4 (for answering user questions)
        self.system_prompt = """
        You are an AI assistant that analyzes videos through natural language captions of video frames.
        You will be given captions from relevant frames that match the user's question.
        Use these captions to provide detailed, accurate responses about the video content.
        
        Guidelines:
        1. Base your responses on the actual caption content provided.
        2. Be specific about what the captions describe.
        3. Use confident language when describing clear observations.
        4. Express uncertainty when appropriate.
        5. Connect observations across multiple captions when relevant.
        6. Describe actions, movements, and techniques in detail.
        7. Include context about the environment and setting.
        
        Remember: The captions are generated from actual video frames, so your responses
        should reflect what is genuinely described in these captions.
        """
    
    def chat(self, user_message: str) -> str:
        """
        Process the user's message, retrieve the most relevant captions,
        and generate a GPT-4 based response.
        """
        # Retrieve relevant captions from the FAISS index
        results = query_captions(user_message, self.index, self.captions, top_k=3)
        context_lines = [f"Frame caption: {cap} (relevance: {score:.2f})" for cap, score in results]
        context = "\n".join(context_lines)
        
        # Create the GPT prompt using the retrieved caption context
        prompt = (
            f"The following captions were generated from relevant video frames:\n\n{context}\n\n"
            f"Based on these frame descriptions, please answer the following question:\n"
            f"Question: {user_message}\n\n"
            f"Answer:"
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            assistant_message = response.choices[0].message.content.strip()
            return assistant_message
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return error_msg

def main():
    try:
        # Replace with your video path and OpenAI API key
        video_path = "En-Garde+Lunge - Made with Clipchamp.mp4"
        openai_api_key = "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"  # Replace with your actual OpenAI API key
        
        logger.info("\nInitializing VideoChat with RAG-based captioning and semantic search...")
        video_chat = VideoChat(video_path, openai_api_key, frame_interval=3, focus_topic="fencing")
        
        logger.info("\nVideoChat initialized successfully!")
        logger.info("Ask questions about the video (type 'quit' to exit)")
        logger.info("\nExample questions:")
        logger.info("- What's happening in this video?")
        logger.info("- Can you describe the fencing techniques shown?")
        logger.info("- What equipment or gear do you see?")
        logger.info("- How many people are in the scene?")
        logger.info("- What's the environment like?")
        
        while True:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit']:
                break
            if user_input:
                response = video_chat.chat(user_input)
                print(f"\nAssistant: {response}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
