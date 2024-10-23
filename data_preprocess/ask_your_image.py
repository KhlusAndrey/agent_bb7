from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration,
)

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

import base64
import requests
import os
import dotenv

_ = dotenv.load_dotenv(dotenv.find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def blip_process_image(image: Image.Image, image_id: str) -> tuple[str, str]:
    """Process image using blip model"""
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    text = "Describe very well this image"
    inputs = blip_processor(image, text, return_tensors="pt")
    out = blip_model.generate(**inputs)
    description = blip_processor.decode(out[0], skip_special_tokens=True)

    print(f"Processed image: {image_id}")
    print(f"Generated description: {description}")

    return (image_id, description)


def deplot_process_image(image: Image.Image, image_id: str) -> tuple[str, str]:
    """Process image using deplot model"""
    deplot_processor = Pix2StructProcessor.from_pretrained("google/deplot")
    deplot_model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")

    text = "Generate underlying data table of the figure below:"
    inputs = deplot_processor(image, text, return_tensors="pt")
    out = deplot_model.generate(**inputs)
    description = deplot_processor.decode(out[0], skip_special_tokens=True)

    print(f"Processed image: {image_id}")
    print(f"Generated description: {description}")

    return (image_id, description)


def llava_describe_image(image: Image, prompt: str):
    """Describe image and get analytical description using llava model"""
    image_tensor = llava_process_image(image)
    prompt, conv = create_llava_prompt(prompt)

    MODEL = "4bit/llava-v1.5-13b-3GB"
    model_name = get_model_name_from_path(MODEL)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True
    )

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria(
        keywords=[stop_str], tokenizer=tokenizer, input_ids=input_ids
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.01,
            max_new_tokens=512,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )
    return tokenizer.decode(
        output_ids[0, input_ids.shape[1] :], skip_special_tokens=True
    ).strip()


def llava_process_image(image: Image.Image, image_id: str) -> tuple[str, torch.Tensor]:
    """Process image using llava model"""
    MODEL = "4bit/llava-v1.5-13b-3GB"
    model_name = get_model_name_from_path(MODEL)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=MODEL, model_base=None, model_name=model_name, load_4bit=True
    )
    args = {"image_aspect_ratio": "pad"}
    image_tensor = process_images([image], image_processor, args)
    return (image_id, image_tensor.to(model.device, dtype=torch.float16))


def create_llava_prompt(prompt: str) -> str:
    """Create llava prompt"""
    CONV_MODE = "llava_v0"
    conv = conv_templates[CONV_MODE].copy()
    roles = conv.roles
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv.append_message(roles[0], prompt)
    conv.append_message(roles[1], None)
    return conv.get_prompt()


def openai_describe_image(image_path: str, prompt: str):
    """Describe image and get analytical description using openai model"""
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


if __name__ == "__main__":
    image = Image.open("/home/wsl/brandbastion/charts/chart5_page_0.jpg")
    prompt = "Describe this image in analytical terms with concepts data and keywords"
    print(llava_describe_image(image, prompt))

    response = openai_describe_image(
        "/home/wsl/brandbastion/charts/chart5_page_0.jpg", prompt
    )
    print(response.json().get("choices")[0].get("message").get("content"))
