import os
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

def download_movenet_thunder():
    # Specify the model URL for MoveNet Thunder
    model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
    
    print("Downloading MoveNet Thunder model...")
    model = hub.load(model_url)
    movenet_thunder = model.signatures['serving_default']
    
    # Create a models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    tf.saved_model.save(model, 'models/movenet_thunder')
    print("MoveNet Thunder model downloaded and saved to models/movenet_thunder")
    
    return movenet_thunder

def test_movenet(model):
    # Create a simple test with a blank image (using correct size for Thunder)
    test_image = np.zeros((256, 256, 3), dtype=np.int32)  # Changed to int32
    input_image = tf.convert_to_tensor(test_image)
    input_image = tf.expand_dims(input_image, axis=0)
    
    # Run inference
    try:
        outputs = model(input_image)
        print("Model test successful!")
        return True
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("Setting up MoveNet Thunder...")
    
    # Check if CUDA is available
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    try:
        # Download and save the model
        model = download_movenet_thunder()
        
        # Test the model
        test_result = test_movenet(model)
        
        if test_result:
            print("\nMoveNet Thunder setup completed successfully!")
            print("The model is saved in the 'models/movenet_thunder' directory")
        else:
            print("\nMoveNet Thunder setup completed but model test failed.")
            print("Please check your TensorFlow installation and try again.")
            
    except Exception as e:
        print(f"\nError during setup: {str(e)}")
        print("Please make sure you have TensorFlow 2.x and tensorflow-hub installed.")
        print("You can install them using:")
        print("pip install tensorflow tensorflow-hub opencv-python numpy") 