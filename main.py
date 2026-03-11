from src.etl import build_data_warehouse
import os 
from dotenv import load_dotenv

load_dotenv()
root_dir = os.getenv("PROJECT_ROOT")

def main():
    build_data_warehouse(root_dir)


if __name__ == "__main__":
    main()
