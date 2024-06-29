import logging
import os
from datetime import datetime

# Define log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Define logs directory path within the project structure
logs_dir = os.path.join(os.getcwd(), "logs")

# Create logs directory if it doesn't exist
os.makedirs(logs_dir, exist_ok=True)

# Define full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Example usage to demonstrate logging
if __name__ == "__main__":
    logging.info("Logging setup complete and ready for use.")