from fastapi import APIRouter, UploadFile, File , HTTPException, Request
from pypdf import PdfReader
from pypdf.errors import EmptyFileError, PdfReadError, PdfStreamError

from ai_document_intelligence.services.document_service import ingest_document
from pathlib import Path

from ai_document_intelligence.schemas.document import AskQuestionRequest, AskQuestionResponse
from ai_document_intelligence.services.retrieval_service import search_documents,document_exists
from ai_document_intelligence.services.llm_service import generate_answer
from ai_document_intelligence.services.file_service import save_uploaded_file
from ai_document_intelligence.exceptions import (
    DocumentNotFoundError,
    FileTooLargeError,
    InvalidFileTypeError
)


router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

@router.post("")
def upload_document(request: Request,file: UploadFile = File(...)):
    # Here you would handle the uploaded file, e.g., save it, process it, etc.

    temp_file_path = None  # Initialize temp_file_path to None

    try:
        temp_file_path = save_uploaded_file(file)

        try:
            reader = PdfReader(temp_file_path, strict=True)
            len(reader.pages)

            document_id = ingest_document(temp_file_path,request.app.state.vector_store)
        except (EmptyFileError, PdfReadError, PdfStreamError):
            raise HTTPException(
                status_code=400,
                detail="Invalid or corrupted PDF file."
            )
        except ValueError as ve:
            if str(ve) == "No text found in the PDF.":
                raise HTTPException(
                    status_code=400,
                    detail="The uploaded PDF contains no text."
                ) from ve
            raise

        return {
            "document_id": document_id,
            "filename": file.filename,
            "message": "Document uploaded and processed successfully."
        }
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail=str(exc)
        ) from exc
    except InvalidFileTypeError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc
    finally:
        # Clean up the temporary file
        if temp_file_path is not None:
            Path(temp_file_path).unlink(missing_ok=True)
        file.file.close()  # Close the file after processing

@router.post("/{document_id}/ask", response_model=AskQuestionResponse)
def ask_question(document_id: str, request:Request, payload: AskQuestionRequest):

    if not document_exists(document_id,request.app.state.vector_store):
        raise DocumentNotFoundError(document_id)

    search_results = search_documents(
        document_id=document_id,
        query=payload.question,
        top_k=3,
        vector_store=request.app.state.vector_store
    )

    if not search_results:
        return {
            "document_id": document_id,
            "question": payload.question,
            "answer": "I don't know.",
            "retrieved_chunks": 0,
            "sources": []
        }

    answer = generate_answer(
        question=payload.question,
        search_results=search_results
    )

    sources = [
        {
            "page": result["metadata"]["page_label"],
            "text": result["content"],
            "score": result["score"]
        }
        for result in search_results
    ]

    response = AskQuestionResponse(
        document_id=document_id,
        question=payload.question,
        answer=answer,
        retrieved_chunks=len(search_results),
        sources=sources,
    )
    return response

