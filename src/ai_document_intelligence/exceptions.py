class FileTooLargeError(Exception):
    pass


class InvalidFileTypeError(Exception):
    pass

class DocumentNotFoundError(Exception):
    """Raised when a requested document does not exist."""

    def __init__(self, document_id: str):
        self.document_id = document_id
        super().__init__(f"Document not found: {document_id}")