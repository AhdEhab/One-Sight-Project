from PIL import Image
import numpy as np
import re
import easyocr
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
import io

def arabic_text_from_image(image_input):
    # ------------------ 🔤 Helper: Check Arabic Only ------------------
    def is_arabic_only(text):
        return all(
            re.fullmatch(r'[\u0600-\u06FF]', ch) or ch.isspace()
            for ch in text
        )

    # ------------------ 🖼️ Helper: Describe image in Arabic ------------------
    def describe_image_arabic(pil_image):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        inputs = processor(images=pil_image, return_tensors="pt")
        out = model.generate(**inputs)
        english_description = processor.decode(out[0], skip_special_tokens=True)

        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        translator = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        translated = tokenizer(english_description, return_tensors="pt", padding=True)
        output = translator.generate(**translated)
        arabic_description = tokenizer.decode(output[0], skip_special_tokens=True)
        return arabic_description

    # ------------------ 📥 Load image ------------------
    if isinstance(image_input, (str, bytes)):  
        img = Image.open(image_input).convert("RGB")
    else:  
        img = Image.open(io.BytesIO(image_input.read())).convert("RGB")

    # ------------------ 🔍 OCR ------------------
    reader = easyocr.Reader(['ar', 'en'], gpu=False)
    results = reader.readtext(np.array(img))  

    # ------------------ 🧹 Filter Arabic only ------------------
    filtered_results = [res for res in results if is_arabic_only(res[1])]

    # ------------------ 📏 Group lines ------------------
    lines = []
    for res in filtered_results:
        (tl, tr, br, bl), text = res[0], res[1]
        y_center = (tl[1] + bl[1]) / 2
        height = abs(bl[1] - tl[1])
        x_start = tl[0]

        matched = False
        for line in lines:
            avg_y = sum([w['y_center'] for w in line]) / len(line)
            if abs(y_center - avg_y) < height * 0.8:
                line.append({'x': x_start, 'text': text, 'y_center': y_center})
                matched = True
                break
        if not matched:
            lines.append([{'x': x_start, 'text': text, 'y_center': y_center}])

    ordered_lines = sorted(
        [sorted(line, key=lambda w: w['x'], reverse=True) for line in lines],
        key=lambda line: sum([w['y_center'] for w in line]) / len(line)
    )

    final_text_lines = [" ".join([w['text'] for w in line]) for line in ordered_lines]
    cleaned_lines = [line for line in final_text_lines if len(line.split()) >= 2]

    if cleaned_lines:
        return "\n".join(cleaned_lines)
    else:
        arabic_description = describe_image_arabic(img)
        return arabic_description
