import os
import time
import torch
import torch.nn.functional as F
import statistics
from transformers import CLIPProcessor, CLIPModel

def analyze_descriptors(model, processor, video_features, descriptors, device):
    """
    Compute normalized similarity between a batch of text descriptors and the given video features.
    Returns a list of (descriptor, similarity) pairs.
    """
    # Prepare input text as a batch
    inputs = processor(text=descriptors, return_tensors="pt", padding=True)
    # Move input tensors to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute text features in one forward pass
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    # Normalize text features
    text_features = F.normalize(text_features, p=2, dim=1)

    # Compute similarity with the video features
    # video_features is a single feature vector, so expand it to batch dimension
    video_features_expanded = video_features.unsqueeze(0).expand(len(descriptors), -1)
    # Dot products along the feature dimension
    sims = torch.sum(text_features * video_features_expanded, dim=1)

    # Move back to CPU if needed for easy Python processing
    sims = sims.cpu().numpy()

    # Pair each descriptor with its similarity score
    return list(zip(descriptors, sims))

def get_raw_recognition(model, processor, video_features, device):
    """Get raw recognition of what's in the video, plus aggregated stats."""
    # Large set of diverse descriptors to check against
    raw_descriptors = [
        # General scene descriptors
        "indoor scene", "outdoor scene", "bright lighting", "dark environment",
        # People and actions
        "person", "multiple people", "standing", "moving", "walking", "running",
        "jumping", "training", "exercising", "teaching", "learning",
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
        "technical demonstration", "skill display", "practice routine"
    ]
    
    print("\n=== Baseline Video Content Analysis ===")
    print("Analyzing what CLIP recognizes most strongly in the video...")

    # Normalize video features
    video_features = F.normalize(video_features, p=2, dim=0).to(device)

    start_time = time.time()
    # Compute similarities in a single batch
    similarities = analyze_descriptors(model, processor, video_features, raw_descriptors, device)
    elapsed_time = time.time() - start_time
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Collect raw scores for stats
    scores = [sim for (_, sim) in similarities]
    mean_score = statistics.mean(scores)
    std_score = statistics.pstdev(scores)  # population stdev
    min_score = min(scores)
    max_score = max(scores)

    # Print some summary stats
    print("\n--- Summary Statistics ---")
    print(f"Number of descriptors: {len(raw_descriptors)}")
    print(f"Analysis time:         {elapsed_time:.2f} seconds")
    print(f"Mean similarity:       {mean_score:.3f}")
    print(f"Std Dev (similarity):  {std_score:.3f}")
    print(f"Min similarity:        {min_score:.3f}")
    print(f"Max similarity:        {max_score:.3f}")

    # How many descriptors exceed certain common thresholds?
    thresholds = [0.2, 0.25, 0.3]
    for t in thresholds:
        count_above = sum(score > t for score in scores)
        print(f"Descriptors > {t:.2f} : {count_above} / {len(scores)}")

    # Print top 15 descriptors
    print("\n--- Top 15 descriptors ---")
    for i, (descriptor, score) in enumerate(similarities[:15], 1):
        confidence = ("Very High" if score > 0.3 else
                      "High" if score > 0.25 else
                      "Medium" if score > 0.2 else "Low")
        print(f"{i:2d}. {descriptor:<25} - Score: {score:.3f} ({confidence})")

    # Print bottom 5 descriptors
    print("\n--- Bottom 5 descriptors ---")
    for descriptor, score in similarities[-5:]:
        confidence = ("Very High" if score > 0.3 else
                      "High" if score > 0.25 else
                      "Medium" if score > 0.2 else "Low")
        print(f"    {descriptor:<25} - Score: {score:.3f} ({confidence})")

def main():
    # Determine if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CLIP model
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    # Load video features
    print("\nLoading video features...")
    video_features = torch.load("data/local_video/clip_features.pt", map_location=device)

    # Show raw recognition + summary stats
    get_raw_recognition(model, processor, video_features, device)
    
    print("\n=== Interactive Query Mode ===")
    print("Ask specific questions about what's in the video.")
    print("Examples:")
    print("- Is there someone wearing protective gear?")
    print("- Are they performing any specific movements?")
    print("- What kind of environment is this filmed in?")
    print("Type 'quit' to exit.\n")
    
    # Normalize the video features just once
    video_features = F.normalize(video_features, p=2, dim=0)

    while True:
        query = input("What would you like to know about the video? ").strip()
        if query.lower() in ['quit', 'exit']:
            break
            
        if not query:
            continue

        # Process text query
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        
        # Get text features
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=1).squeeze()

        # Compute similarity
        similarity = torch.dot(text_features, video_features.to(device))
        score = similarity.item()
        
        print(f"\nConfidence Level: {score:.3f}")
        if score > 0.3:
            print("✓✓ Very High Confidence: This is definitely in the video!")
        elif score > 0.25:
            print("✓ High Confidence: This is very likely in the video")
        elif score > 0.2:
            print("? Medium Confidence: This might be in the video")
        else:
            print("× Low Confidence: Probably not in the video")

if __name__ == "__main__":
    main()
