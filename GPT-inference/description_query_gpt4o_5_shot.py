import json
import os
import requests
import base64

api_key= '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'

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

prompts = {
    "prompt1": "Please describe the image. Your description should include is there any people, construction equipment, or material stockpiles in the image, where are they approximately located in the image, and how many are they. Please describe the activities. Please also include any outstanding construction elements in the image.", 
    "prompt2": "Please describe the image. Try your best to find the number, location, and activities of people, and construction equipment. Also, find the type and number of material stockpiles in the image. Also describe anything unusual in the image.",
    "prompt3": "Please describe the image. Try your best to find the number, location, and activities of people, and construction equipment. And the type and number of material stockpiles in the image. Also describe anything unusual in the image. Please describe with only one paragraph.",
    "prompt4": "Please describe the image. Your description should include the number, location, and activities of people, construction equipment, or material stockpiles in the image. Please describe with only one paragraph.",
    "prompt5": "Please describe the image. Your description should include the number, location, and activities of people, construction equipment, or material stockpiles in the image. Do not make assumptions and be concise. Please describe with only one paragraph."
}

# Replace 'your_script.py' with the actual name of the Python script you want to execute
image_folder = "/home/xuezheng/Desktop/VLM/10k_images"
data_split_id = "random1"

# 5-shot Prompts
# Few shot examples are: Image ID: 0000015, 0000020, 0000039, 0000041, 0000046 
_PROMPT_SYSTEM = """You are a construction site inspector. You are responsible for viewing the given image and give helpful, detailed, and polite answers to your supervisor."""

_PROMPT_USER_FEWSHOT_1 = """
Your task is to describe the image. I will give you a few image-description pairs as examples first.
"""

# Reason: rich info, mid distance
_EXAMPLE_FEWSHOT_0000015 = """
Many bundles of rebars are on the ground. Some portion of the rebars is covered with blue cloth. 
A worker with a yellow hard hat and a safety vest at the left is processing rebars with a machine. There is a shed in the image. 
There are two other workers with yellow hard hats and safety vest on the right of the image. There is an unfinished concrete building at the background.  
"""
image_0000015 = encode_image(os.path.join(image_folder, '0000015.jpg'))

# Reason: rich info, long distance
_EXAMPLE_FEWSHOT_0000020 = """
A busy construction site. There are about 33 workers with red or yellow hard hats and safety vests. 
There are two excavators on the hill in the background. A tower crane is lifting a bundle of rebars for the workers. On the right, there is a pile driver in the background. There are some wood boards and rebars scattered at the site.
"""
image_0000020 = encode_image(os.path.join(image_folder, '0000020.jpg'))

# Reason: rich info, long distance, long discription
_EXAMPLE_FEWSHOT_0000039 = """
A dump truck and dirt piles are in the foreground of the image. On the left, there are three bulldozers, a roller, and 
a lot of orange tubes. An excavator is lifting a pile of orange tubes. In the middle of the image, there is a giant crawler crane. A pile of black pipes, and mint-colored pipes. on the right of the image, there is a water truck, a bulldozer, and a pile of black pipes. In the background, there are two tower cranes. 
"""
image_0000039 = encode_image(os.path.join(image_folder, '0000039.jpg'))

# Reason: rich info, short distance
_EXAMPLE_FEWSHOT_0000041 = """
Six workers are pouring concrete with a hose at a construction site.
"""
image_0000041 = encode_image(os.path.join(image_folder, '0000041.jpg'))

# Reason: rich info, short distance, night
_EXAMPLE_FEWSHOT_0000046 = """
There are eight workers with yellow hard hats at an unfinished tunnel-like structure. One of them is standing on top of the finished portion of the tunnel. There is a flash light in the background. 
"""
image_0000046 = encode_image(os.path.join(image_folder, '0000046.jpg'))

def main():

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
            
    storage_file = "gpt4o_5_shot_" + data_split_id + "_" + "prompt5" + "_" + "seed1" + ".json"
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

    # Loop through all examples
    for image in image_list_to_finish:
        print(f"Running {i}/{tot}")
        print(image)
        i += 1

        image_path = os.path.join(image_folder, image + '.jpg')
        base64_image = encode_image(image_path)
        instance_id = image

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
        "model": "gpt-4o",
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 1024,
        "messages": [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": _PROMPT_SYSTEM
                },
                {
                "type": "text",
                "text": _PROMPT_USER_FEWSHOT_1
                },
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_0000015}",
                    "detail": "high"
                }
                },
                {
                "type": "text",
                "text": _EXAMPLE_FEWSHOT_0000015
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_0000020}",
                    "detail": "high"
                }
                },
                {
                "type": "text",
                "text": _EXAMPLE_FEWSHOT_0000020
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_0000039}",
                    "detail": "high"
                }
                },
                {
                "type": "text",
                "text": _EXAMPLE_FEWSHOT_0000039
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_0000041}",
                    "detail": "high"
                }
                },
                {
                "type": "text",
                "text": _EXAMPLE_FEWSHOT_0000041
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_0000046}",
                    "detail": "high"
                }
                },
                {
                "type": "text",
                "text": _EXAMPLE_FEWSHOT_0000046
                },
                # {
                # "type": "text",
                # "text": "How many images you have received so far, including all?"
                # },
                {
                "type": "text",
                "text": prompts['prompt5']
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
                }
            ]
            }
        ]
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print(response.json())
        cleared_reply = response.json()['choices'][0]['message']['content']

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