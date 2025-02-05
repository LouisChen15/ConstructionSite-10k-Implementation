import json
import os
from predict import Predictor

prompts = {
    "prompt1": "Please describe the image. Your description should include is there any people, construction equipment, or material stockpiles in the image, where are they approximately located in the image, and how many are they. Please describe the activities. Please also include any outstanding construction elements in the image.", 
    "prompt2": "Please describe the image. Try your best to find the number, location, and activities of people, and construction equipment. Also, find the type and number of material stockpiles in the image. Also describe anything unusual in the image.",
    "prompt3": "Please describe the image. Try your best to find the number, location, and activities of people, and construction equipment. And the type and number of material stockpiles in the image. Also describe anything unusual in the image. Please describe with only one paragraph.",
    "prompt4": "Please describe the image. Your description should focus on the number, location, and activities of people, construction equipment, or material stockpiles in the image. Do not make assumptions and be concise. Please describe with only one paragraph.",
    "prompt5": "Please describe the image. Your description should include the number, location, and activities of people, construction equipment, or material stockpiles in the image. Do not make assumptions, be concise, and describe facts only. Please describe with only one paragraph."
}

SIMPROMPT = "Please describe the given image with only one paragraph."

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

    predictor = Predictor()

    #for prompt_id, user_input in prompts.items():
    for seed_num in range(5):
        seed_num += 1
        seed_str = str(seed_num)
        
        storage_file = "minigpt4_v2_7B_5_shot_" + data_split_id + "_" + "prompt5" + "_" + "seed" + seed_str + ".json"
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
            instance_id = image
            output = predictor.predict(image_path=image_path, prompt=prompts['prompt5'])
            
            cleared_reply = output.replace('\n\n\n', '').strip()

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
