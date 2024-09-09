from PIL import Image, ImageEnhance
import pytesseract
import requests
from io import BytesIO
import os

# Set Tesseract's path to the executable
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Update this path if needed

# Define the path to the Tesseract tessdata folder
tessdata_dir = './tessdata'  # Update this path if different

# Ensure language data files are present; adjust languages as needed
languages = 'spa+eng+jpn+nep+jpn_vert'  # Spanish language code

def preprocess_image(image):
    """
    Preprocess the image to improve OCR accuracy.
    """
    # Convert to grayscale
    image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    
    # Binarize the image
    threshold = 128
    image = image.point(lambda p: p > threshold and 255)
    
    return image

def extract_text_from_image(image_source):
    """
    Extracts text from an image. Supports both local file paths and URLs.
    """
    try:
        # Check if the image_source is a URL
        if image_source.startswith('http://') or image_source.startswith('https://'):
            # Fetch the image from the URL
            response = requests.get(image_source)
            response.raise_for_status()  # Check if the request was successful
            image = Image.open(BytesIO(response.content))
        else:
            # Load the image from a local file path
            image = Image.open(image_source)
        
        # Preprocess the image
        image = preprocess_image(image)
        
        # Extract text using Spanish language
        extracted_text = pytesseract.image_to_string(image, lang=languages, config=f'--tessdata-dir {tessdata_dir}')
        return extracted_text
    
    except requests.exceptions.RequestException as e:
        return f"Network error: {e}"
    except IOError as e:
        return f"File error: {e}"
    except Exception as e:
        return f"Error extracting text: {e}"

# Example usage
# image_path = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSqaojtloJPbvWJWeFWDBOomUpKlVqAZz5Z4Q&s' 
# image_path = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR9QUEaLpQIRATe9BGVToaBlmeBPHynZa_wHw&s'
# image_path = 'https://www.clipartmax.com/png/middle/195-1950892_restroom-sign-japan-japanese-sign-in-japan.png'
image_path = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQpb_WfGGU7aPQv17kqRtXEIm3uwctMepYIBA&s'
extracted_text = extract_text_from_image(image_path)


print("Extracted Text:")
print(extracted_text)
