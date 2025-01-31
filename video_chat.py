import os
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from openai import OpenAI
from typing import List, Tuple
import json

class VideoChat:
    def __init__(self, clip_features_path: str, system_prompt: str = None):
        """Initialize the VideoChat system."""
        # Load CLIP model for processing new queries
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        
        # Initialize OpenAI client with API key
        self.client = OpenAI(
            api_key="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"  # Replace with your actual API key from platform.openai.com
        )
        
        # Load pre-extracted video features
        self.video_features = torch.load(clip_features_path)
        self.video_features = F.normalize(self.video_features, p=2, dim=0)
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Set up system prompt
        self.system_prompt = system_prompt or """You are a friendly and helpful AI assistant who can see and understand videos. You're having a casual conversation with a user about what you see in the video.

        You have access to similarity scores that indicate how well different descriptions match the video content. While you use these scores to understand what's in the video, you should speak naturally without mentioning the technical details or scores.

        Guidelines for your responses:
        1. Be conversational and friendly, like you're chatting with a friend
        2. Don't mention scores, CLIP, or technical terms
        3. Express confidence when you see something clearly
        4. Use phrases like "I can see...", "It looks like...", "I notice..." when describing what you observe
        5. Be honest about uncertainty - use phrases like "I'm not entirely sure, but..." or "It's a bit hard to tell..."
        6. Keep the context of the conversation flowing naturally
        
        Remember: You're having a friendly chat about what you see in the video, not giving a technical analysis."""
        
    def _get_clip_similarity(self, query: str) -> float:
        """Get CLIP similarity score for a text query."""
        inputs = self.processor(text=[query], return_tensors="pt", padding=True)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=1).squeeze()
            
        similarity = torch.dot(text_features, self.video_features).item()
        return similarity
    
    def _analyze_content(self, query: str) -> List[Tuple[str, float]]:
        """Analyze video content with related queries to provide context."""
        # Generate related aspects to check based on the query
        related_queries = [
            query,
            f"video showing {query}",
            f"scene with {query}",
            f"person doing {query}",
            "indoor scene",
            "outdoor scene",
            "multiple people",
            "single person",
            "sports or exercise",
            "training or practice",
            "fencing",
            "martial arts",
            "teaching",
            "demonstration",
            "equipment",
            "protective gear",
            "movement",
            "action",
            "technique"
        ]
        
        # Get similarity scores for all related queries
        similarities = [(q, self._get_clip_similarity(q)) for q in related_queries]
        return sorted(similarities, key=lambda x: x[1], reverse=True)
    
    def chat(self, user_message: str) -> str:
        """Process user message and generate a response."""
        # Analyze video content related to the query
        content_analysis = self._analyze_content(user_message)
        
        # Format content analysis for GPT in a more natural way
        high_confidence = [(q, s) for q, s in content_analysis if s > 0.25]
        medium_confidence = [(q, s) for q, s in content_analysis if 0.2 <= s <= 0.25]
        
        content_info = "Here's what I can see in the video:\n"
        if high_confidence:
            content_info += "\nClearly visible elements:\n"
            content_info += "\n".join([f"- {q}" for q, _ in high_confidence])
        
        if medium_confidence:
            content_info += "\n\nPossible or partial elements:\n"
            content_info += "\n".join([f"- {q}" for q, _ in medium_confidence])
        
        # Prepare messages for GPT
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": content_info}
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add user's current message
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Get response from GPT using new API format
            response = self.client.chat.completions.create(
                model="gpt-4",  # or "gpt-3.5-turbo" for a faster, cheaper option
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            # Extract response
            assistant_message = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

def main():
    # Initialize VideoChat
    video_chat = VideoChat("data/local_video/clip_features.pt")
    
    print("Welcome to VideoChat! Ask me anything about the video (type 'quit' to exit)")
    print("Example questions:")
    print("- What's happening in the video?")
    print("- Is this an indoor or outdoor scene?")
    print("- What kind of activities are shown?")
    print("- How many people are in the video?")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
            
        if user_input:
            response = video_chat.chat(user_input)
            print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main() 