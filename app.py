import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import requests
import transformers
# Load CLIP model and preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
from transformers import AutoTokenizer, AutoModelForCausalLM

MIN_TRANSFORMERS_VERSION = '4.25.1'

# check transformers version
assert transformers.__version__ >= MIN_TRANSFORMERS_VERSION, f'Please upgrade transformers to version {MIN_TRANSFORMERS_VERSION} or higher.'

# init
tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Chat-3B-v1", torch_dtype=torch.bfloat16)

# Function to predict descriptions and probabilities
def predict(news,vocab):
    prompt = f"<human>: Rephrase {news} without changing the structure, but use these vocabulary {vocab}\n<bot>:"
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.7, top_k=50, return_dict_in_generate=True
    )
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)
    return output_str

# Streamlit app
def main():
    st.title("Todays News")

    # Instructions for the user
    st.markdown("---")
    st.markdown("### Vocabulary Requested")

    # Upload image through Streamlit with a unique key
    # uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"], key="uploaded_image")
    gist_url = 'https://gist.githubusercontent.com/nafisdev/381aac7ee98f6e4311ae90798f308956/raw/035681ab33e53cbd09c30c6cb59827d06cc8bd72/vocab.txt'


    response = requests.get(gist_url)
    gist_content = response.text
    # link = 'https://newsdata.io/api/1/archive?apikey=pub_480828ed4d09aafac2c6eb519f41a0715d7c8&q=USAlocal&language=en&from_date=2024-07-05&to_date=2024-07-06'
    # res = requests.get(link)

    # Display the content in Streamlit
    st.title("Text from GitHub Gist")
    st.text(gist_content)
    text ='In a hidden forest glade, a solitary fox dances under the moonlight, surrounded by fireflies. Nearby, an ancient tree whispers secrets to the wind, while a curious owl watches from its perch. The night is alive with the sounds of nature, weaving a mysterious and enchanting tapestry of life.'
    out =  predict(text,gist_content)
    st.write(f"**Todays Paragraph** {out}")
#     if st.button("Predict"):
#         if all(descriptions):
#             # Make predictions
#             best_description, best_prob = predict(pil_image, descriptions)

#             # Display the highest probability description and its probability
#             st.write(f"**Best Description:** {best_description}")
#             st.write(f"**Prediction Probability:** {best_prob:.2%}")

#             # Display progress bar for the highest probability
#             st.progress(float(best_prob))

    # if uploaded_image is not None:
    #     # Convert the uploaded image to PIL Image
    #     # pil_image = Image.open(uploaded_image)

    #     # Limit the height of the displayed image to 400px
    #     st.image(pil_image, caption="Uploaded Image.", use_column_width=True, width=200)
        
    #     # Instructions for the user
    #     st.markdown("### 2 Lies and 1 Truth")
    #     st.markdown("Write 3 descriptions about the image, 1 must be true.")

    #     # Get user input for descriptions
    #     description1 = st.text_input("Description 1:", placeholder='A red apple')
    #     description2 = st.text_input("Description 2:", placeholder='A car parked in a garage')
    #     description3 = st.text_input("Description 3:", placeholder='An orange fruit on a tree')

    #     descriptions = [description1, description2, description3]

    #     # Button to trigger prediction
    #     if st.button("Predict"):
    #         if all(descriptions):
    #             # Make predictions
    #             best_description, best_prob = predict(pil_image, descriptions)

    #             # Display the highest probability description and its probability
    #             st.write(f"**Best Description:** {best_description}")
    #             st.write(f"**Prediction Probability:** {best_prob:.2%}")

    #             # Display progress bar for the highest probability
    #             st.progress(float(best_prob))

if __name__ == "__main__":
    main()
