import json
import os
import requests
import base64
from retry import retry
from google import genai
from google.genai import types

api_key= 'Your Own API Key'

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as f:
      return f.read()

image_folder = "../10k_images"
data_split_id = "random1"


SYSTEM_PROMPT = """
You are a construction site safety inspector. You are responsible for viewing the given image and give helpful, detailed, and polite answers to your supervisor. You only answer questions that are asked by the supervisor and in the exact way as requested.
"""

FEW_SHOT_PROMPT = """
You will be asked to read the image and identify violations of safety rules that appears in the image. You also need to provide a short reasoning and bounding boxs showing the location of the violation. 

I will give you five examples to show you how your answer should be formatted and what your reasoning should include.
"""

USER_PROMPT = """
Please read the image and identify if there are violations of the following four safety rules in the image, do not include violations that do not exist in your answer, assume no violation if the visual information is not enough to make a judgement:

1. Use of basic PPE when on foot at construction sites. Machine operators do not need PPE. (hard hats, properly worn clothes covering shoulders and legs, shoes that can cover toes, high-visibility retroreflective vests at night, face shield or safety glasses when cutting, welding, grinding, or drilling).

2. Use of safety harness when working from a height of three meters and the edges are without any edge protection.

3. Adoption of edge protection or edge warning including guardrails, fences, for underground projects three meters in depth with steep retaining wall and for human to stand.

4. Appearance of worker in the blind spots of the operator and within the operation radius of excavators in operation, or excavators with operators inside.

Your answer should be in the format of {"id of the safety rule": {"reason": one or two sentences explaining who violate the rule in the image and the specific reason, "bounding box": [the location of violation in the image x_min, y_min, x_max, y_max in 0-1 normalized space]}}.

Return {"0": "No violations"} if you find no violation in the image.
"""

EXAMPLE_PROMPT_0000001 = """
Example:

{"0": "No violations"}
"""
image_0000001 = encode_image(os.path.join(image_folder, '0000001.jpg'))

EXAMPLE_PROMPT_0000007 = """
Example:

{"1": {"reason": "Multiple workers not wearing hard hats nor high-visibility vests working at night.", "bounding_box": [0.14, 0.09, 1.0, 0.66]}, "3": {"reason": "Opening not protected on both the left and the right of the images.", "bounding_box": [0.0, 0.53, 0.46, 0.99]}}
"""
image_0000007 = encode_image(os.path.join(image_folder, '0000007.jpg'))

EXAMPLE_PROMPT_0000019 = """
Example:

{"1": {"reason": "Worker with a black cap and white shirt on the left is not wearing a hard hat.", "bounding_box": [0.27, 0.47, 0.42, 0.68]}}
"""
image_0000019 = encode_image(os.path.join(image_folder, '0000019.jpg'))

EXAMPLE_PROMPT_0000327 = """
Example:

{"4": {"reason": "The worker holding an umbrella is too close to the excavator is operation.", "bounding_box": [0.25, 0.28, 0.95, 0.76]}}
"""
image_0000327 = encode_image(os.path.join(image_folder, '0000327.jpg'))

EXAMPLE_PROMPT_0004235 = """
Example:

{"1": {"reason": "None of the workers wear a hard hat. The worker on the ground level in the middle is not wearing his shirt properly while the worker on top of the scaffold wears a sleeveless shirt.", "bounding_box": 0.02, 0.01, 0.44, 0.47]}, "2": {"reason": "The worker standing on the scaffold does not have a safety harness.", "bounding_box": [0.01, 0.04, 0.25, 0.48]}}
""" 
image_0004235 = encode_image(os.path.join(image_folder, '0004235.jpg'))


@retry((KeyError), tries = 5, delay = 10, jitter=5)
def main():

    with open("../Annotations/dataset10k_metadata.json", "r") as file:
        data_split = json.load(file)
    test_split = data_split['test_split']

    image_list = test_split
    image_list = sorted(image_list)
    tot = len(image_list)
            
    storage_file = "gemini_5_shot_" + data_split_id + "_" + "vqa" + "_" + "seed1" + ".json"
    print(storage_file)

    try:
        with open(storage_file, 'r') as file:
            data = json.load(file)
            finished_list = list(data.keys())
    except:
        finished_list = []
    
    image_list_to_finish = [image_id for image_id in image_list if image_id not in finished_list]

    instance_id_list = []
    reply_list = []
    i = 1+len(finished_list)
    cached_token=0
    prompt_token=0
    candidates_token=0

    # Loop through all examples
    for image in image_list_to_finish:
        print(f"Running {i}/{tot}")
        print(image)
        i += 1

        image_path = os.path.join(image_folder, image + '.jpg')
        image_bytes = encode_image(image_path)
        instance_id = image

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,   
                temperature=0.2,
                top_p=1.0,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
            contents=[
                # === Few-shot examples ===
                FEW_SHOT_PROMPT,
                types.Part.from_bytes(
                    data=image_0000001,  
                    mime_type="image/jpeg",
                ),
                EXAMPLE_PROMPT_0000001,

                types.Part.from_bytes(
                    data=image_0000007,
                    mime_type="image/jpeg",
                ),
                EXAMPLE_PROMPT_0000007,

                types.Part.from_bytes(
                    data=image_0000019,
                    mime_type="image/jpeg",
                ),
                EXAMPLE_PROMPT_0000019,

                types.Part.from_bytes(
                    data=image_0000327,
                    mime_type="image/jpeg",
                ),
                EXAMPLE_PROMPT_0000327,

                types.Part.from_bytes(
                    data=image_0004235,
                    mime_type="image/jpeg",
                ),
                EXAMPLE_PROMPT_0004235,

                # === User request ===
                USER_PROMPT,
                types.Part.from_bytes(
                    data=image_bytes,    
                    mime_type="image/jpeg",
                ),
            ],
        )

        if response.usage_metadata.cached_content_token_count is None:
            cached_token += 0
            prompt_token += response.usage_metadata.prompt_token_count
        else:
            cached_token += response.usage_metadata.cached_content_token_count
            prompt_token += (response.usage_metadata.prompt_token_count - response.usage_metadata.cached_content_token_count)
        candidates_token += response.usage_metadata.candidates_token_count
        print(cached_token)
        print(prompt_token)
        print(candidates_token)
        cleared_reply = response.text

        instance_id_list.append(instance_id)
        reply_list.append(cleared_reply)

        # intermediate save
        if len(instance_id_list) == 1:
            answer = dict(zip(instance_id_list, reply_list))
            print(answer)
            instance_id_list = []
            reply_list = []
            if not os.path.exists(storage_file):
                with open(storage_file, 'w') as file: 
                    json.dump(answer, file)
            else:
                with open(storage_file, 'r') as file: 
                    data = json.load(file)
                    answer.update(data)
                with open(storage_file, "w") as file:
                    json.dump(answer, file)

    # Write and sort the result again outside the loop
    answer = dict(zip(instance_id_list, reply_list))

    with open(storage_file, 'r') as file: 
        data = json.load(file)
        answer.update(data)

    sorted_dict = dict(sorted(answer.items(), key=lambda item: int(item[0])))  

    with open(storage_file, "w") as file:
        json.dump(sorted_dict, file)


if __name__ == "__main__":
    main()