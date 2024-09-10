import pytesseract
from PIL import Image
import os

def image_to_text(image_path, output_path):
    # Open the image using PIL
    image = Image.open(image_path)
    
    # Use pytesseract to do OCR on the image
    text = pytesseract.image_to_string(image)
    
    # Write the extracted text to a file
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)
    
    print(f"Text extracted and saved to {output_path}")

# Example usage
image_path = 'lab.jpg'  # Replace with the path to the uploaded image
output_path = 'medical_report.txt'  # The path where you want to save the text file

image_to_text(image_path, output_path)