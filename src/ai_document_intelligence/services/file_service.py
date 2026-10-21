import tempfile
from pathlib import Path
from fastapi import UploadFile
from ai_document_intelligence.exceptions import FileTooLargeError, InvalidFileTypeError

MAX_FILE_SIZE = 10 * 1024 * 1024
CHUNK_SIZE = 1024 * 1024


def save_uploaded_file(file: UploadFile) -> Path:
    """Save an uploaded PDF to a temporary file."""
    if file.content_type not in ["application/pdf"]:
        raise InvalidFileTypeError("Invalid or corrupted PDF file.")

    temp_file_path = None  # Initialize temp_file_path to None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file_path = Path(temp_file.name)

            total_size = 0

            while True:
                chunk = file.file.read(CHUNK_SIZE)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise FileTooLargeError("File size exceeds the 10 MiB limit.")

                temp_file.write(chunk)
    except Exception:
        if temp_file_path is not None:
            temp_file_path.unlink(missing_ok=True)
        raise

    return temp_file_path