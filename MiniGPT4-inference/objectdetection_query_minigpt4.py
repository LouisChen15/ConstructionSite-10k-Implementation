import json
import os
from predict import Predictor

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

    predictor = Predictor()

    #for prompt_id, user_input in prompts.items():
    for seed_num in range(3):
        seed_num += 1
        seed_str = str(seed_num)
        
        storage_file = "minigpt4_v2_7B_" + data_split_id + "_" + object_to_detect + "_" + "seed" + seed_str + ".json"
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
            output = predictor.predict(image_path=image_path, prompt=_PROMPT_USER)

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
