import json
import os
import requests
import base64


api_key= "YOUR_OWN_API_KEY"

prompts = {
    "prompt1": "Please describe the image. Your description should include is there any people, construction equipment, or material stockpiles in the image, where are they approximately located in the image, and how many are they. Please describe the activities. Please also include any outstanding construction elements in the image.", 
    "prompt2": "Please describe the image. Try your best to find the number, location, and activities of people, and construction equipment. Also, find the type and number of material stockpiles in the image. Also describe anything unusual in the image.",
    "prompt3": "Please describe the image. Try your best to find the number, location, and activities of people, and construction equipment. And the type and number of material stockpiles in the image. Also describe anything unusual in the image. Please describe with only one paragraph.",
    "prompt4": "Please describe the image. Your description should include the number, location, and activities of people, construction equipment, or material stockpiles in the image. Please describe with only one paragraph.",
    "prompt5": "Please describe the image. Your description should include the number, location, and activities of people, construction equipment, or material stockpiles in the image. Do not make assumptions and be concise. Please describe with only one paragraph."
}

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def main():

    image_folder = "10k_images"
    data_split_id = "random1"

    with open("dataset10k_metadata.json", "r") as file:
        data_split = json.load(file)
    test_split = data_split

    image_list = test_split
    image_list = sorted(image_list)
    tot = len(image_list)


    # for prompt_id, user_input in prompts.items():
    #     for seed_num in range(5):
    #         seed_num += 1
    #         seed_str = str(seed_num)
            
    storage_file = data_split_id + "_" + "prompt5" + "_" + "seed1" + ".json"
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
        "model": "gpt-4-1106-vision-preview",
        "temperature": 0.2,
        "top_p": 1.0,
        "max_tokens": 1024,
        "messages": [
            {
            "role": "system",
            "content": "You are a construction site inspector. You are responsible for viewing the given image and give helpful, detailed, and polite answers to your supervisor."
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Please describe the image. Your description should include the number, location, and activities of people, construction equipment, or material stockpiles in the image. Do not make assumptions, be concise, and describe facts only. Please describe with only one paragraph."
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