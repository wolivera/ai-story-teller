from transformers import BlipProcessor, BlipForConditionalGeneration

def get_image_description(raw_image):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    image_description = processor.decode(out[0], skip_special_tokens=True)
    print("Image description: ", image_description)

    return image_description
