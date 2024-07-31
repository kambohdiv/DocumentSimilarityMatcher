# Invoice Similarity and Feature Extraction

## Overview

This project demonstrates how to extract text from PDF invoices, compute similarity between invoices using cosine similarity, and extract specific features from the invoices using regular expressions.

## Concepts

### 1. PDF Text Extraction

**Objective:** Extract text from PDF files.

- **Library Used:** `PyPDF2`
- **Function:** `extractText(pdf_path)`
  
  This function reads a PDF file and extracts its text content. It handles exceptions that might occur during file reading.

### 2. TF-IDF Vectorization

**Objective:** Convert text documents into numerical vectors to measure their similarity.

- **Library Used:** `sklearn.feature_extraction.text.TfidfVectorizer`
- **Function:** `calculateSimilarity(text1, text2)`

  The `TfidfVectorizer` transforms the text into TF-IDF vectors, which are then used to compute the cosine similarity between two text documents.

### 3. Cosine Similarity

**Objective:** Measure the similarity between two vectors.

- **Library Used:** `sklearn.metrics.pairwise.cosine_similarity`
- **Function:** `calculateSimilarity(text1, text2)`

  Cosine similarity calculates the cosine of the angle between two vectors. It provides a measure of similarity ranging from -1 (completely dissimilar) to 1 (completely similar).

### 4. Feature Extraction

**Objective:** Extract specific features from text using regular expressions.

- **Library Used:** `re`
- **Function:** `extractFeatures(text)`

  This function uses regular expressions to extract specific features from the invoice text, such as invoice number, date, and amount.

### 5. Invoice Database Management

**Objective:** Manage a collection of invoices and find the most similar one.

- **Class:** `InvoiceDatabase`
- **Methods:**
  - `addInvoice(invoice)`: Adds an invoice to the database.
  - `findMostSimilar(inputInvoice)`: Finds the most similar invoice to the given input invoice by comparing their text similarity.
