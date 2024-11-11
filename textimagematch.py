import streamlit as st
import requests
import os
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration
import openai

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Set up OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai_api_key"]

# Define the GitHub API URL to list image files in the folder
GITHUB_API_URL = "https://api.github.com/repos/scooter7/textimagematch/contents/images"
GITHUB_RAW_URL = "https://raw.githubusercontent.com/scooter7/textimagematch/main/images/"

# Function to list image filenames from GitHub folder
def list_image_files_from_github():
    response = requests.get(GITHUB_API_URL)
    if response.status_code == 200:
        files = response.json()
        image_files = [file['name'] for file in files if file['name'].lower().endswith(('.jpg', '.jpeg', '.png'))]
        return image_files
    else:
        st.error("Failed to fetch image list from GitHub.")
        return []

# Function to fetch image from GitHub
def fetch_image_from_github(image_name):
    url = f"{GITHUB_RAW_URL}{image_name}"
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return image
    else:
        st.error(f"Failed to fetch image {image_name} from GitHub.")
        return None

# Function to generate image description (prevent hashing the image argument)
@st.cache_resource
def generate_image_description(_image):
    inputs = processor(images=_image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# Function to evaluate the best image match
def evaluate_best_match(user_text, image_descriptions):
    messages = [
        {"role": "system", "content": "You are an assistant that matches user text to the best image based on descriptions."},
        {"role": "user", "content": f"Given the user's text: '{user_text}', choose the best matching image from the following descriptions and provide a rationale: {image_descriptions}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # Correct syntax for using gpt-4o-mini
        messages=messages,
        max_tokens=150
    )
    
    # Debug: Output the entire response to check the format
    st.write("API Response:", response)

    # Access the message content as an attribute
    try:
        return response.choices[0].message.content.strip()
    except (AttributeError, IndexError):
        st.error("Unexpected response format from OpenAI.")
        return "Could not retrieve a valid response."

# Streamlit app
st.title("Image-Text Matching AI")
st.write("This app will find the best image to pair with your text from a folder of images based on AI-generated descriptions.")

# User input text field
user_text = st.text_input("Enter the text you want to pair with an image:")

if user_text:
    # Get the list of image filenames from the GitHub folder
    image_filenames = list_image_files_from_github()

    if image_filenames:
        # Generate descriptions for each image and display them
        image_descriptions = []
        for image_name in image_filenames:
            image = fetch_image_from_github(image_name)
            if image:
                description = generate_image_description(image)
                if description:
                    image_descriptions.append(f"Image {image_name}: {description}")
                    st.image(image, caption=f"{image_name} - {description}")
                    st.write(f"Description for {image_name}: {description}")  # Debug line to verify description
                else:
                    st.warning(f"No description generated for {image_name}.")
            else:
                st.error(f"Failed to fetch image {image_name}.")

        # Query the best match using OpenAI
        if image_descriptions:
            rationale = evaluate_best_match(user_text, image_descriptions)
            st.write("### AI's Best Match and Rationale:")
            st.write(rationale)
        else:
            st.error("No image descriptions generated.")
    else:
        st.error("No images found in the GitHub folder.")
