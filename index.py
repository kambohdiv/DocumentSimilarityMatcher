# Import PdfReader for reading PDF files
from PyPDF2 import PdfReader   # type: ignore
# Import TfidfVectorizer for converting text to TF-IDF vectors
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
 # Import cosine_similarity for computing similarity between vectors
from sklearn.metrics.pairwise import cosine_similarity   # type: ignore
 # Import re for regular expression operations
import re 

def extractText(pdf_path):
    """
    Extracts text from a PDF file.
    
    Args:
    pdf_path (str): The file path of the PDF.

    Returns:
    str: The extracted text from the PDF.
    """
    text = ""
    try:
        # Open the PDF file in read-binary mode
        with open(pdf_path, "rb") as file:
            reader = PdfReader(file)  # Create a PdfReader object
            for page in reader.pages:
                # Extract text from each page and append to text variable
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")  # Handle and report errors
    return text

def extractFeatures(text):
    """
    Extracts specific features from the text using regular expressions.
    
    Args:
    text (str): The text from which to extract features.

    Returns:
    dict: A dictionary containing extracted features such as invoice number, date, and amount.
    """
    features = {}
    features['invoice_number'] = re.search(r'Invoice Number:\s*(\S+)', text)
    features['date'] = re.search(r'Date:\s*(\d{4}-\d{2}-\d{2})', text)
    features['amount'] = re.search(r'Amount:\s*\$([\d,\.]+)', text)
    return features

def calculateSimilarity(text1, text2):
    """
    Calculates the cosine similarity between two text documents.
    
    Args:
    text1 (str): The first text document.
    text2 (str): The second text document.

    Returns:
    float: The cosine similarity score between the two text documents.
    """
    vectorizer = TfidfVectorizer()  # Create a TF-IDF vectorizer
    vectors = vectorizer.fit_transform([text1, text2])  # Transform texts into TF-IDF vectors
    return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]  # Compute cosine similarity

class InvoiceDatabase:
    """
    A class to manage a collection of invoices and find the most similar invoice.
    """
    def __init__(self):
        self.invoices = []  # Initialize an empty list to store invoices

    def addInvoice(self, invoice):
        """
        Adds an invoice to the database.
        
        Args:
        invoice (dict): A dictionary containing the invoice text and filename.
        """
        self.invoices.append(invoice)  # Append the invoice to the list

    def findMostSimilar(self, inputInvoice):
        """
        Finds the most similar invoice to a given input invoice.
        
        Args:
        inputInvoice (dict): A dictionary containing the input invoice text and filename.

        Returns:
        tuple: A tuple containing the most similar invoice and its similarity score.
        """
        maxSimilarity = 0  # Initialize the maximum similarity score
        mostSimilarInvoice = None  # Initialize the most similar invoice
        for invoice in self.invoices:
            if invoice['filename'] != inputInvoice['filename']:  # Skip comparing with itself
                similarity = calculateSimilarity(inputInvoice['text'], invoice['text'])  # Calculate similarity
                if similarity > maxSimilarity:  # Check if the similarity is the highest so far
                    maxSimilarity = similarity
                    mostSimilarInvoice = invoice
        return mostSimilarInvoice, maxSimilarity  # Return the most similar invoice and its similarity score

if __name__ == "__main__":
    db = InvoiceDatabase()  # Create an instance of InvoiceDatabase

    # List of PDF files to be processed
    files = [
        'train/2024.03.15_0954.pdf',
        'train/2024.03.15_1145.pdf',
        'train/Faller_8.PDF',
        'train/invoice_77073.pdf',
        'train/invoice_102856.pdf'
    ]

    # Process each PDF file
    for pdfFile in files:
        try:
            invoice_text = extractText(pdfFile)  # Extract text from the PDF
            db.addInvoice({'text': invoice_text, 'filename': pdfFile})  # Add the invoice to the database
            print(f"Successfully processed {pdfFile}")
        except Exception as e:
            print(f"Error processing {pdfFile}: {str(e)}")  # Handle and report errors

    # Choose one file as the input invoice (for example)
    inputFile = 'test/invoice_102857.pdf'
    inputInvoiceText = extractText(inputFile)  # Extract text from the input invoice
    inputInvoice = {'text': inputInvoiceText, 'filename': inputFile}  # Create an invoice dictionary

    # Find the most similar invoice
    mostSimilar, similarityScore = db.findMostSimilar(inputInvoice)
    if mostSimilar:
        print(f"Most similar invoice: {mostSimilar['filename']}")
        print(f"Similarity score: {similarityScore}")
    else:
        print("No similar invoice found.")

    # Extract features from the input invoice
    input_features = extractFeatures(inputInvoiceText)
    print("\nInput Invoice Features:")
    for key, value in input_features.items():
        if value:
            print(f"{key}: {value.group(1)}")  # Print the extracted feature if found
        else:
            print(f"{key}: Not found")  # Print 'Not found' if the feature was not extracted
