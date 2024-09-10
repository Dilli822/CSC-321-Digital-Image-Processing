import spacy
import medspacy

# Load the MedSpaCy model
nlp = spacy.load("en_core_sci_md")  # SciSpaCy model suitable for medical text

def extract_medical_terms(text):
    # Process the text using MedSpaCy NLP pipeline
    doc = nlp(text)
    
    # Extract entities recognized as medical terms
    medical_terms = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "CONDITION", "MEDICATION"]]
    
    return medical_terms

def validate_user_terms(user_terms, extracted_terms):
    valid_terms = []
    invalid_terms = []
    
    for term in user_terms:
        if term.lower() in [t.lower() for t in extracted_terms]:
            valid_terms.append(term)
        else:
            invalid_terms.append(term)
    
    return valid_terms, invalid_terms

# Sample text extracted from a medical report
sample_text = """
The patient shows elevated cholesterol levels and hypertension. 
Blood pressure readings are consistently high, indicating a risk of cardiovascular disease.
"""

# Extract medical terms from the sample text
extracted_terms = extract_medical_terms(sample_text)
print("Medical Terms Found:", extracted_terms)

# User input and validation
while True:
    user_input = input("\nEnter medical terms to validate (comma-separated), or 'q' to quit: ")
    if user_input.lower() == 'q':
        break
    
    user_terms = [term.strip() for term in user_input.split(',')]
    valid, invalid = validate_user_terms(user_terms, extracted_terms)
    
    print("Valid terms:", valid)
    print("Invalid terms:", invalid)

print("Thank you for using the Medical Term Extractor and Validator!")