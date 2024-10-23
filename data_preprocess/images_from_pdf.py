import fitz
import io
from PIL import Image
from data_preprocess.ask_your_image import blip_process_image


def extract_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)

    images = []

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img_data = pix.tobytes()
        image = Image.open(io.BytesIO(img_data))
        images.append(image)

    return images

def process_imges_from_pdf(pdf_path: str):
    images = extract_images_from_pdf(pdf_path)
    for i, image in enumerate(images):
        blip_process_image(image, f"{pdf_path}_page_{i}")
        

if __name__ == "__main__":
    process_imges_from_pdf("/home/wsl/brandbastion/charts/chart5.pdf")
