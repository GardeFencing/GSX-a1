import os
import time
import torch
import torch.nn.functional as F
import statistics
from transformers import CLIPProcessor, CLIPModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_descriptors(model, processor, video_features, descriptors, device):
    """
    Compute normalized similarity between a batch of text descriptors and the given video features.
    Returns a list of (descriptor, similarity) pairs.
    """
    inputs = processor(text=descriptors, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = F.normalize(text_features, p=2, dim=1)
    # Expand the single video feature to match the number of descriptors
    video_features_expanded = video_features.unsqueeze(0).expand(len(descriptors), -1)
    sims = torch.sum(text_features * video_features_expanded, dim=1)
    sims = sims.cpu().numpy()
    return list(zip(descriptors, sims))

def get_raw_recognition(model, processor, video_features, device):
    """Get raw recognition of what's in the video, plus aggregated stats."""
    raw_descriptors = [
        # General scene descriptors
        "indoor scene", "outdoor scene", "bright lighting", "dim lighting",
        # People and actions
        "person", "multiple people", "crowd", "group of people", "solitary person",
        "standing", "sitting", "moving", "walking", "running", "jumping",
        "exercising", "training", "teaching", "learning",
        # Sports and activities
        "sports", "martial arts", "fencing", "boxing", "training session",
        "practice", "demonstration", "competition", "workout", "exercise",
        # Equipment and clothing
        "sports equipment", "protective gear", "uniform", "white clothing",
        "athletic wear", "sports gear", "mask", "helmet", "protective equipment",
        # Environment
        "gym", "training room", "studio", "sports facility", "indoor facility",
        "practice area", "training center", "fitness center",
        # Video style
        "professional video", "amateur video", "instructional video",
        "training video", "educational content", "demonstration video",
        # Motion and technique
        "fast movement", "slow movement", "precise movement", "controlled motion",
        "technical demonstration", "skill display", "practice routine",
        # High-energy descriptors
        "dynamic", "action-packed", "high-energy", "intense", "focused"
    ]
    
    logger.info("=== Baseline Video Content Analysis ===")
    logger.info("Analyzing what CLIP recognizes most strongly in the video...")
    
    video_features = F.normalize(video_features, p=2, dim=0).to(device)
    start_time = time.time()
    similarities = analyze_descriptors(model, processor, video_features, raw_descriptors, device)
    elapsed_time = time.time() - start_time
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    scores = [sim for (_, sim) in similarities]
    mean_score = statistics.mean(scores)
    std_score = statistics.pstdev(scores)  # population stdev
    min_score = min(scores)
    max_score = max(scores)
    
    logger.info("----- Summary Statistics -----")
    logger.info(f"Number of descriptors: {len(raw_descriptors)}")
    logger.info(f"Analysis time:         {elapsed_time:.2f} seconds")
    logger.info(f"Mean similarity:       {mean_score:.3f}")
    logger.info(f"Std Dev (similarity):  {std_score:.3f}")
    logger.info(f"Min similarity:        {min_score:.3f}")
    logger.info(f"Max similarity:        {max_score:.3f}")
    
    for t in [0.2, 0.25, 0.3]:
        count_above = sum(score > t for score in scores)
        logger.info(f"Descriptors > {t:.2f} : {count_above} / {len(scores)}")
    
    logger.info("----- Top 15 Descriptors -----")
    for i, (descriptor, score) in enumerate(similarities[:15], 1):
        confidence = ("Very High" if score > 0.3 else
                      "High" if score > 0.25 else
                      "Medium" if score > 0.2 else "Low")
        logger.info(f"{i:2d}. {descriptor:<25} - Score: {score:.3f} ({confidence})")
    
    logger.info("----- Bottom 5 Descriptors -----")
    for descriptor, score in similarities[-5:]:
        confidence = ("Very High" if score > 0.3 else
                      "High" if score > 0.25 else
                      "Medium" if score > 0.2 else "Low")
        logger.info(f"    {descriptor:<25} - Score: {score:.3f} ({confidence})")

def interactive_query_mode(model, processor, video_features, device):
    """Allow the user to query the video interactively and print confidence levels."""
    logger.info("=== Interactive Query Mode ===")
    logger.info("Ask specific questions about what's in the video. Type 'quit' to exit.")
    video_features = F.normalize(video_features, p=2, dim=0)
    
    while True:
        query = input("What would you like to know about the video? ").strip()
        if query.lower() in ['quit', 'exit']:
            break
        if not query:
            continue
        
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=1).squeeze()
        
        similarity = torch.dot(text_features, video_features.to(device)).item()
        print(f"\nConfidence Level: {similarity:.3f}")
        if similarity > 0.3:
            print("✓✓ Very High Confidence: This is definitely in the video!")
        elif similarity > 0.25:
            print("✓ High Confidence: This is very likely in the video")
        elif similarity > 0.2:
            print("? Medium Confidence: This might be in the video")
        else:
            print("× Low Confidence: Probably not in the video")

def main():
    # Determine if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load CLIP model (using base variant for consistency)
    logger.info("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    # Load video features
    logger.info("\nLoading video features...")
    video_features = torch.load("data/local_video/clip_features.pt", map_location=device)
    
    # Show raw recognition + summary stats
    get_raw_recognition(model, processor, video_features, device)
    
    logger.info("\n=== Interactive Query Mode ===")
    logger.info("Ask specific questions about what's in the video.")
    logger.info("Examples:")
    logger.info("- Is there someone wearing protective gear?")
    logger.info("- Are they performing any specific movements?")
    logger.info("- What kind of environment is this filmed in?")
    logger.info("Type 'quit' to exit.\n")
    
    interactive_query_mode(model, processor, video_features, device)

if __name__ == "__main__":
    main()
