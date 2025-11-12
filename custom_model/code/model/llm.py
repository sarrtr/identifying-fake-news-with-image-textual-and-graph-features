import base64
import os
import requests
from dotenv import load_dotenv

load_dotenv()

image_dir_path = '/repo/project_deepfake/project/datasets/fakeddit_dataset/images_sampled/'

proxies = {
    "http": os.getenv('PROXY_HTTP'),
    "https": os.getenv('PROXY_HTTP')
}

token = os.getenv("GROK_TOKEN")

url = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = image_dir_path+'1a1dsi.jpg'

def get_image_llm_explanation(image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    data = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image? Explain what you see"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "temperature": 0.5,
        "max_tokens": 256
    }

    response = requests.post(url, headers=headers, json=data, proxies=proxies)

    if response.status_code == 413:
        print('Maximum allowed size for a request containing a base64 encoded image is 4MB')
        return 
    elif response.status_code == 200:
        response = response.json()
        try:
            return response['choices'][0]['message']['content']
        except:
           return 
    else:
        print(response.json())
        return 