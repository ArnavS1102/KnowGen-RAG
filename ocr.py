from preprocess_pdf import PDF2MD
from postprocess_pdf import Cleaner, extract_last_itemize_block
import os

if __name__ == "__main__":
    source_dir = os.getenv("PDF_FOLDER")
    dest_dir = os.getenv("MD_FOLDER")

    pdf2md = PDF2MD()
    cleaner = Cleaner()

    pdf2md.parse_dir(dest_dir, source_dir)
    cleaner.clean_files(dest_dir)

