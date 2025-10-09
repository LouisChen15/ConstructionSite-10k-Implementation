import json
import os
import requests
import base64
from retry import retry
from google import genai
from google.genai import types

api_key= 'AIzaSyCQ8Z1j7X4yp9FtV-T4ii-xHLdlOaFvB4Y'

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as f:
      return f.read()

_PROMPT_SYSTEM = """You are a construction site safety inspector." "You are responsible for viewing the given image and give helpful, and polite answers to your supervisor." "You only answer questions that are asked by the supervisor and in the exact way as requested."""

object_to_detect = "excavator"

_PROMPT_USER = f"""
Please detect all instances of {object_to_detect} in the image.

Your answer should be include [the location of {object_to_detect} in the image in x_min, y_min, x_max, y_max in 0-1 normalized space], there may be more than one instance in each image.

Return ["None"] if you find no {object_to_detect} in the image.
"""

def main():

    # Replace 'your_script.py' with the actual name of the Python script you want to execute
    image_folder = "/home/xuezheng/Desktop/VLM/10k_images"
    data_split_id = "random1"

    with open("/home/xuezheng/Desktop/VLM/Annotations/dataset10k_metadata.json", "r") as file:
        data_split = json.load(file)
    test_split = data_split['test_split']

    image_list = test_split
    image_list = sorted(image_list)
    tot = len(image_list)


    # for prompt_id, user_input in prompts.items():
    #     for seed_num in range(5):
    #         seed_num += 1
    #         seed_str = str(seed_num)
            
    storage_file = "gemini_" + data_split_id + "_" + object_to_detect + "_" + "seed1" + ".json"
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
                system_instruction=_PROMPT_SYSTEM,   
                temperature=0.2,
                top_p=1.0,
                max_output_tokens=1024,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
            contents=[
                _PROMPT_USER,
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
        cleared_reply = response.text.replace("`", "").replace("\n", " ").replace("'", " ").replace("json", " ")

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