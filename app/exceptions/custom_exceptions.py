
"""Custom exceptions for Talk2SQL."""

import traceback
from pathlib import Path


class CustomException(Exception):
    """Exception enriched with the original error location."""

    def __init__(
        self,
        error_message: str,
        error_details: Exception | None = None,
    ) -> None:
        self.original_message = error_message
        self.file_name = "Unknown"
        self.line_number = 0

        # Extract location information from the original exception.
        if error_details is not None:
            tb = error_details.__traceback__

            if tb is not None:
                frames = traceback.extract_tb(tb)
                if frames:
                    last_frame = frames[-1]
                    self.file_name = Path(
                        last_frame.filename
                    ).name
                    self.line_number = last_frame.lineno

        detailed_message = (
            f"Error occurred in file: {self.file_name}\n"
            f"Line number: {self.line_number}\n"
            f"Error message: {self.original_message}"
        )

        super().__init__(detailed_message)
