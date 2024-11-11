import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai
import glob

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Set up OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

# Define the GitHub URL where images are stored
GITHUB_FOLDER_URL = "https://github.com/scooter7/textimagematch/raw/main/images"

# Function to fetch image from GitHub
def fetch_image_from_github(image_name):
    url = f"{GITHUB_FOLDER_URL}/{image_name}"
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        st.error(f"Failed to fetch image {image_name} from GitHub.")
        return None

# Function to generate image description
@st.cache_resource
def generate_image_description(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# Function to evaluate the best image match
def evaluate_best_match(user_text, image_descriptions):
    prompt = f"Given the user's text: '{user_text}', choose the best matching image from the following descriptions and provide a rationale: {image_descriptions}"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response["choices"][0]["text"].strip()

# Streamlit app
st.title("Image-Text Matching AI")
st.write("This app will find the best image to pair with your text from a folder of images based on AI-generated descriptions.")

# User input text field
user_text = st.text_input("Enter the text you want to pair with an image:")

if user_text:
    # Simulate fetching a list of image filenames (for demonstration purposes)
    image_filenames = ["image1.jpg", "image2.png", "image3.jpeg"]  # Replace with actual filenames from GitHub

    # Generate descriptions for each image and display them
    image_descriptions = []
    for image_name in image_filenames:
        image = fetch_image_from_github(image_name)
        if image:
            description = generate_image_description(image)
            if description:
                image_descriptions.append(f"Image {image_name}: {description}")
                st.image(image, caption=f"{image_name} - {description}")
            else:
                st.warning(f"No description generated for {image_name}.")

    # Query the best match using OpenAI
    if image_descriptions:
        rationale = evaluate_best_match(user_text, image_descriptions)
        st.write("### AI's Best Match and Rationale:")
        st.write(rationale)
    else:
        st.error("No image descriptions generated.")
