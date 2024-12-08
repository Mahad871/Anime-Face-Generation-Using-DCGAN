import os
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.models import load_model

# Global generator model
generator = None

def load_generator_model():
    global generator
    try:
        model_path = "./saved_models/complete_generator2_model.h5"
        generator = load_model(model_path)
        print("Generator model loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def generate_anime_image(btn=None):  # Add parameter to handle Gradio button input
    """Generate a single anime image"""
    global generator
    
    if generator is None:
        if not load_generator_model():
            return np.zeros((64, 64, 3))
    
    try:
        noise = tf.random.normal([1, 100])
        generated_image = generator(noise, training=False)
        generated_image = generated_image[0].numpy()
        generated_image = np.clip(generated_image * 255, 0, 255).astype(np.uint8)
        return generated_image
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return np.zeros((64, 64, 3))

# Load model on startup
load_generator_model()

# Create Gradio Interface
interface = gr.Interface(
    fn=generate_anime_image,
    inputs=[],
    outputs=gr.Image(),
    title="Anime Face Generator",
    description="Let's Generate Anime Images with Magic (AI)",
    allow_flagging="never"
)

# Launch the interface
interface.launch(share=True, debug=True)