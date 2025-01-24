# Azure-Based Corrosion Detection and Reporting Application

## Overview
This Django-based application provides an end-to-end workflow for detecting corrosion on images using machine learning models, storing results in Azure, and generating detailed reports. It supports various functionalities, including user authentication, image uploads, prediction workflows, and integration with Azure Blob Storage and Cosmos DB for data storage and retrieval.

---

## Features
- **Corrosion Detection**:
  - Detect corrosion in chips and bearings using pre-trained ONNX models.
  - Generate metrics such as defect percentage and defect size.

- **Azure Integration**:
  - Store processed images and metadata in Azure Blob Storage.
  - Save detection results and metadata in Azure Cosmos DB.

- **Data Transformation**:
  - Convert masks to sparse matrices for efficient storage.
  - Support base64 encoding/decoding for images and masks.

- **User Interface**:
  - Upload images via web interface.
  - View and analyze processed images in a user-friendly gallery.

- **Report Generation**:
  - Generate detailed reports in `.docx` format with custom templates.
  - Export data to CSV for further analysis.

- **Authentication**:
  - User registration and login functionality.
  - User-specific containers in Azure for organized data storage.

---

## Requirements
- Python 3.8 or higher
- Azure Blob Storage and Cosmos DB accounts
- ONNX runtime environment for ML model inference
- Django 4.x or higher

---

## Installation

### Prerequisites
1. Install Python 3.8+.
2. Set up Azure Blob Storage and Cosmos DB.
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
