# Gemini CLI Context for Voice Identification Project

This document provides an overview of the `voice_identification` project, detailing its purpose, technologies, architecture, and instructions for building and running.

## Project Overview

This project is a Python-based voice identification and transcription service, developed using FastAPI. It incorporates various machine learning models for audio processing tasks such as Voice Activity Detection (VAD), speaker diarization (implied by `speaker_tracks` and `separated_regions` in `TranscriptionService`), and transcription utilizing the VinAI PhoWhisper-medium model, specialized for Vietnamese Automatic Speech Recognition (ASR). The service integrates with AWS S3 for video storage and audio extraction, and uses PostgreSQL for data persistence. Its core functionality involves handling file uploads, extracting audio from video, processing audio for speech, and transcribing speech into text with word-level timestamps.

## Main Technologies

*   **Backend Framework:** FastAPI
*   **Web Server:** Uvicorn
*   **Programming Language:** Python
*   **Machine Learning/Audio Processing:** PyTorch, Hugging Face Transformers (`vinai/PhoWhisper-medium`), `pyannote.audio`, `librosa`, `moviepy`.
*   **Cloud Services:** AWS S3 (for file storage)
*   **Database:** PostgreSQL (with SQLAlchemy for ORM)
*   **Dependency Management:** `requirements.txt` (typically pip/pipenv)
*   **Configuration:** `starlette.config` from `.env` files.
*   **Logging:** `loguru`

## Architecture

The application follows a modular architecture with distinct layers for different functionalities:

*   **API Layer:** Handled by FastAPI routers, exposing endpoints for voice identification and related tasks (e.g., file uploads). The main router is found in `app/api/routers/api.py`, which includes `app.api.routers.upload_router` for file upload functionality.
*   **Core Configuration:** Centralized configuration management using `.env` files, defined in `app/core/config.py`.
*   **Infrastructure Layer:** AWS S3 integration for object storage, with components like `app/api/infra/aws/s3/repository/object.py`.
*   **Services Layer:** Contains business logic for audio extraction (`app/api/services/processing/audio_extractor.py`), VAD analysis (`app/api/services/processing/vad_analyzer.py`), transcription (`app/api/services/processing/transcription_service.py`), and other processing tasks.
*   **Data Model Layer:** Manages ML models (e.g., VAD, Whisper) and their loading/caching, with implementations like `app/data_model/implementations/whisper_transcriber.py`.
*   **Database Layer:** PostgreSQL for persistent storage, with session management defined in `app/api/db/session.py`.

## Building and Running

Follow these steps to set up and run the project:

1.  **Install Dependencies:**
    Navigate to the project root and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables:**
    Create a `.env` file in the project root directory. This file will store environment-specific configurations. Refer to `app/core/config.py` for all available configuration options. Populate it with necessary variables, including AWS credentials, Hugging Face token, Gemini API key, database connection details, and other settings.

    Example `.env` structure:
    ```ini
    PROJECT_NAME="BAP Voice Identification"
    DEBUG=True
    ALLOWED_HOSTS="*"
    DEVICE="cuda" # Set to "cuda" if GPU is available, "cpu" otherwise.
    AWS_ACCESS_KEY="your_aws_access_key"
    AWS_SECRET_KEY="your_aws_secret_key"
    AWS_REGION="ap-northeast-1"
    HF_TOKEN="your_huggingface_token" # Required for Hugging Face models like PhoWhisper
    GEMINI_API_KEY="your_gemini_api_key"

    AUDIO_FILE_SEIZE=52428800    # Max audio file size in bytes (e.g., 50MB)
    VIDEO_FILE_SEIZE=1073741824  # Max video file size in bytes (e.g., 1GB)
    MAX_REQUEST_SIZE=1073741824  # Max request body size in bytes (e.g., 1GB)

    S3_PREFIX_AUDIO="uploads/audio"
    S3_PREFIX_VIDEO="uploads/video"
    AUDIO_EXTS="m4a,ogg,mp3"
    VIDEO_EXTS="mp4"

    UPLOAD_TIMEOUT=300   # Upload timeout in seconds (5 minutes)
    REQUEST_TIMEOUT=300  # Request timeout in seconds (5 minutes)

    DB_HOST="localhost"
    DB_PORT=5432
    DB_USER_NAME="postgres"
    DB_PASSWORD="your_db_password"
    DB_DATABASE="BAP_Voice_Identification"
    ```

3.  **Run Database Migrations:**
    This project uses `alembic` for database migrations. You will need to run migrations to set up your database schema.
    ```bash
    # (TODO: Determine the exact alembic command to run migrations.
    # A common command is `alembic upgrade head`, but it might vary based on project setup.)
    ```

4.  **Run the Application:**
    Once dependencies are installed and environment variables are set, you can start the FastAPI application:
    ```bash
    python app/main.py
    ```
    For development with automatic code reloading (useful for local development):
    ```bash
    uvicorn app.main:app --reload
    ```
    The application will typically run on `http://127.0.0.1:8000` (or as configured in your `.env` file).

## Development Conventions

*   **Logging:** The project utilizes `loguru` for comprehensive and structured logging across the application.
*   **Asynchronous Operations:** `asyncio` is extensively used for efficient handling of I/O-bound operations, particularly in services interacting with external resources or performing long-running tasks.
*   **Type Hinting:** The codebase makes extensive use of Python's type hints, enhancing code readability, maintainability, and enabling better static analysis.
*   **Configuration Management:** Environment-specific configurations are managed through `.env` files, loaded via `starlette.config`, ensuring sensitive information is kept out of the codebase.
*   **Error Handling:** Services and API endpoints typically return structured dictionaries containing `success`, `data`, and `error` keys, providing clear status and error messages.
*   **ML Model Management:** Machine learning models are loaded on demand and cached efficiently using a `ModelRegistry`, optimizing resource utilization and startup times.
*   **GPU Acceleration:** The application automatically detects and prioritizes CUDA for GPU acceleration if available, falling back gracefully to CPU processing when a GPU is not present or configured.
*   **Temporary File Handling:** The `tempfile` module is used for secure and efficient management of temporary files generated during audio and video processing workflows, ensuring proper cleanup.
