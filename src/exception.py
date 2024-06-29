import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Capture detailed error message with file name and line number where the exception occurred.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = "Error occurred in script: [{0}] at line: [{1}] with message: [{2}]".format(
        file_name, line_number, str(error)
    )
    return error_message

class CustomException(Exception):
    """
    Custom exception class to handle exceptions in the hospital readmission prediction project.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
        logging.error(self.error_message)  # Log the error message using the project's logger
    
    def __str__(self):
        return self.error_message

# Example usage
if __name__ == "__main__":
    try:
        # Simulate an error for demonstration purposes
        1 / 0
    except Exception as e:
        raise CustomException(e, sys)