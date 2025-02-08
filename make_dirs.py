import os
from dotenv import load_dotenv

load_dotenv(".env")
if __name__ == "__main__":
    md_path = os.getenv("MD_FOLDER")
    pdf_path = os.getenv("PDF_FOLDER")

    if not os.path.exists(md_path):
        os.mkdir(md_path)

    if not os.path.exists(pdf_path):
        os.mkdir(pdf_path)
    


