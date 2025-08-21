import replicate
import base64
import dotenv
import requests
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

dotenv.load_dotenv()

def image_path_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"

def caption_image(sketch_uri):    
    chunks = replicate.run(
    "openai/gpt-4.1-nano",
    input={
        "top_p": 1,
        "prompt": " ",
        "messages": [],
        "image_input": [sketch_uri],
        "temperature": 1,
        "system_prompt": "You will be given a sketch of the face of a person. From that sketch, you have to guess the genre, whether if the person has facial hair (only for male), the person has glasses, if the person is bald.\nThe resulting conclusion must be given as just boolean flags in text. Without any other information or text. Something exactly like:\nis_male=True\nhas_glasses=False\nis_bald=False\nhas_facial_hair = True\n\nthe process is going to be automated and the outputs interpreted with python exec() so dont add anything else or any other text. The code is expecting exactly those 4 bool variables.",
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "max_completion_tokens": 4096
    })
    response = str("".join(chunks)).strip()
    return response

def generate_variation(reconstructed_face_uri, prompt):
    output = replicate.run(
        "black-forest-labs/flux-kontext-max",
        input={
            "prompt": prompt,
            "input_image": reconstructed_face_uri,
            "aspect_ratio": "match_input_image",
            "output_format": "jpg",
            "safety_tolerance": 2,
            "prompt_upsampling": False
        }
    )
    url = output.url
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

def generate_variations(reconstructed_face_uri, sketch_uri):
    predicted_flags = caption_image(sketch_uri)
    print(predicted_flags)
    is_male = has_glasses = is_bald = has_facial_hair = None
    try:
        local_vars = {}
        exec(predicted_flags, {}, local_vars)
        is_male = local_vars.get('is_male', None)
        has_glasses = local_vars.get('has_glasses', None)
        is_bald = local_vars.get('is_bald', None)
        has_facial_hair = local_vars.get('has_facial_hair', None)
    except Exception as e:
        pass
    
    # Define editing prompts for each casuistic
    editing_captions = {}

    # Incognito (all): Change the clothing to a black hoodie sweater. The hoodie is over the person's head. Also, put black sunglasses.
    editing_captions['incognito'] = (
        "Change the person's clothing to a black hoodie sweater. The hoodie is over the person's head, obscuring the hair. "
        "Also, put black sunglasses on the person. Do not change the face or other features."
    )

    # Hair modifications (for all genders)
    if is_bald is False:
        editing_captions['bald'] = (
            "Make the person bald. Remove all visible hair from the head. Do not change any other facial features."
        )
    if is_bald is True:
        editing_captions['add_hair'] = (
            "Add realistic hair to the person's head. Do not change any other features."
        )

    # Male-specific modifications
    if is_male is True:
        # Only-Male Shave: Remove the facial hair
        if has_facial_hair is True:
            editing_captions['shave'] = (
                "Remove all facial hair from the person, making the face clean-shaven. Do not change any other features."
            )
        # Only-Male Facial hair: Add a beard
        if has_facial_hair is False:
            editing_captions['add_beard'] = (
                "Add a realistic beard to the person's face. Do not change any other features."
            )

    # Oldify
    editing_captions['oldify'] = (
        "Make the person look 5 years older."
    )

    # Only glasses people: Remove the glasses
    if has_glasses is True:
        editing_captions['remove_glasses'] = (
            "Remove any glasses from the person's face. Make sure the eyes are clearly visible. Do not change any other features."
        )

    # Only glassless people: Add prescription glasses
    if has_glasses is False:
        editing_captions['add_glasses'] = (
            "Add realistic prescription glasses to the person's face. Do not use sunglasses. Do not change any other features."
        )

    variations = {}

    def generate_variation_safe(key, prompt):
        try:
            print(f"Generating variation for {key}")
            return key, generate_variation(reconstructed_face_uri, prompt)
        except Exception as e:
            return key, None

    with ThreadPoolExecutor(max_workers=len(editing_captions)) as executor:
        futures = [
            executor.submit(generate_variation_safe, key, prompt)
            for key, prompt in editing_captions.items()
        ]
        for future in as_completed(futures):
            key, result = future.result()
            variations[key] = result

    return variations



if __name__ == "__main__":
    from tqdm import tqdm
    import os

    sketch_uri = image_path_to_base64(r"F:\FaceSketch\full\sketch\4bcae0de-48b1-40c2-8b9f-1625a9419c7f.png")
    reconstructed_face_uri = image_path_to_base64(r"G:\My Drive\PythonProjects\SketchRecognition\ip_adapter_output.png")
    variations = generate_variations(reconstructed_face_uri, sketch_uri)

    save_path = "variations"
    os.makedirs(save_path, exist_ok=True)
    for key, variation in tqdm(variations.items(), desc="Saving variations"):
        if variation is not None:
            variation.save(f"{save_path}\{key}.png")
