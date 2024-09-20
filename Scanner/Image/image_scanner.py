import cv2
import numpy as np
import pytesseract
from PIL import Image

def capture_image():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Document (Press SPACE to capture)', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh

def extract_text(image):
    # Configure Tesseract to use multiple languages and recognize various character types
    custom_config = r'--oem 3 --psm 6 -l eng+fra+deu+spa+ita+por+rus+jpn+kor+chi_sim+chi_tra'
    text = pytesseract.image_to_string(Image.fromarray(image), config=custom_config)
    return text

def main():
    print("Position your document in front of the camera and press SPACE to capture.")
    captured_image = capture_image()
    processed_image = preprocess_image(captured_image)
    extracted_text = extract_text(processed_image)
    
    print("\nExtracted Text:")
    print(extracted_text)

    # Save the processed image and extracted text
    cv2.imwrite('processed_document.png', processed_image)
    with open('extracted_text.txt', 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    
    print("\nProcessed image saved as 'processed_document.png'")
    print("Extracted text saved as 'extracted_text.txt'")

if __name__ == "__main__":
    main()