# Emergency Lighting Detection from Construction Blueprints

This project implements an AI Vision pipeline that extracts emergency lighting information from electrical drawings and prepares structured outputs using computer vision and LLMs.

## Features

- Detects emergency lights (shown as shaded rectangular areas on layout drawings)
- Identifies different types of emergency lights (2' X 4' RECESSED LED LUMINAIRE, WALLPACK WITH BUILT-IN PHOTOCELL, etc.)
- Captures bounding boxes and spatial locations of fixtures and nearby text/symbols
- Extracts static content like general notes and lighting schedule tables
- Groups lighting fixtures based on symbols and provides counts

## API Endpoints

### 1. Upload and Trigger Processing

```
POST /blueprints/upload
```

**Purpose**: Upload a PDF and initiate background processing (CV + OCR + LLM)

**Request**:
- `file`: PDF file (multipart/form-data)
- `project_id` (optional): Project grouping identifier

**Response**:
```json
{
  "status": "uploaded",
  "pdf_name": "E2.4.pdf",
  "message": "Processing started in background."
}
```

### 2. Get Processed Result

```
GET /blueprints/result?pdf_name=<pdf_name>
```

**Purpose**: Retrieve the final grouped result for a given PDF name

**Query Param**:
- `pdf_name`: Name of the uploaded PDF (e.g., E2.4.pdf)

**Response (if processing complete)**:
```json
{
  "pdf_name": "E2.4.pdf",
  "status": "complete",
  "result": {
    "A1": {
      "count": 12,
      "description": "2x4 LED Emergency Fixture"
    },
    "A1E": {
      "count": 5,
      "description": "Exit/Emergency Combo Unit"
    },
    "W": {
      "count": 9,
      "description": "Wall-Mounted Emergency LED"
    }
  }
}
```

**Response (if still processing)**:
```json
{
  "pdf_name": "E2.4.pdf",
  "status": "in_progress",
  "message": "Processing is still in progress. Please try again later."
}
```

### 3. Get Annotation Image

```
GET /blueprints/annotation/<pdf_name>
```

**Purpose**: Retrieve the annotated image showing detected emergency lights

**Response**: PNG image with bounding boxes and labels

## Setup Instructions

### Prerequisites

- Python 3.7+
- Tesseract OCR installed on your system
- OpenAI API key (optional, for advanced LLM processing)

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file based on `.env.example` and configure your environment variables
4. Create necessary directories (if not already present):
   ```
   mkdir uploads results annotations
   ```

### Running the Application

```
python app.py
```

The application will start on http://localhost:5000

## Background Processing

The application handles background processing using Python's threading module. When a PDF is uploaded:

1. The file is saved to the `uploads` directory
2. A processing job is created in the database with status "uploaded"
3. A background thread is started to process the PDF
4. The processing includes:
   - Converting PDF to images
   - Detecting emergency lights
   - Extracting static content (notes, schedules)
   - Grouping fixtures by type
   - Creating annotation images
5. The job status is updated to "complete" when finished

## Result Storage and Retrieval

Results are stored in an SQLite database (`emergency_lighting.db`) with the following tables:

1. `processing_jobs`: Tracks the status of each processing job
2. `extracted_data`: Stores the extracted data (detections, notes, table rows)

When a client requests results:
1. The application queries the database for the job status
2. If complete, it returns the structured result
3. If still processing, it returns a status message

## Technologies Used

- Flask: Web framework
- OpenCV: Computer vision for detection
- Tesseract OCR: Text extraction
- PyMuPDF: PDF processing
- SQLite: Database storage
- OpenAI (optional): Advanced LLM processing