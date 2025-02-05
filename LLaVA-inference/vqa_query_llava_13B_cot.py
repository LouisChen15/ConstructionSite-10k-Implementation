import json
import os
from predict_sockeye_13B import Predictor
import random

_PROMPT_SYSTEM = """You are a construction site safety inspector." "You are responsible for viewing the given image and give helpful, and polite answers to your supervisor." "You only answer questions that are asked by the supervisor and in the exact way as requested."""

_PROMPT_USER_RULE1 = """
Please view the image and identify if there are violations of the following safety rule in the image, also you should strictly follow the format guide provided below: 

Safety Rule: Use of basic PPE when on foot at construction sites. Machine operators do not need PPE. (hard hats, properly worn clothes covering shoulders and legs, shoes that can cover toes, high-visibility retroreflective vests when working at night, face shield or safety glasses when cutting, welding, grinding, or drilling).

If there is violation of the rule, return {"yes": {"reason": one or two sentences explaining who violate the rule, the location in the image, and the specific reason, "bounding box": [the location of violation in the image x_min, y_min, x_max, y_max in 0-1 normalized space]}}. 
Return {"no": "No violations"} if you find no violation in the image.
"""

_PROMPT_USER_RULE2 = """
Please view the image and identify if there are violations of the following safety rule in the image, also you should strictly follow the format guide provided below:

Safety Rule: Use of safety harness when working from a height of three meters and the edges are without any edge protection.

If there is violation of the rule, return {"yes": {"reason": one or two sentences explaining who violate the rule, the location in the image, and the specific reason, "bounding box": [the location of violation in the image x_min, y_min, x_max, y_max in 0-1 normalized space]}}. 
Return {"no": "No violations"} if you find no violation in the image.
"""

_PROMPT_USER_RULE3 = """
Please view the image and identify if there are violations of the following safety rule in the image, also you should strictly follow the format guide provided below:

Safety Rule: Adoption of edge protection or edge warning including guardrails, fences, for underground projects three meters in depth with steep retaining wall and for human to stand.

If there is violation of the rule, return {"yes": {"reason": one or two sentences explaining who violate the rule, the location in the image, and the specific reason, "bounding box": [the location of violation in the image x_min, y_min, x_max, y_max in 0-1 normalized space]}}. 
Return {"no": "No violations"} if you find no violation in the image.
"""

_PROMPT_USER_RULE4 = """
Please view the image and identify if there are violations of the following safety rule in the image, also you should strictly follow the format guide provided below:

Safety Rule: Appearance of worker in the blind spots of the operator and within the operation radius of excavators in operation, or excavators with operators inside. The blind spot of the excavator is the right, the rear, and the rear left of an excavator. 

If there is violation of the rule, return {"yes": {"reason": one or two sentences explaining who violate the rule, the location in the image, and the specific reason, "bounding box": [the location of violation in the image x_min, y_min, x_max, y_max in 0-1 normalized space]}}. 
Return {"no": "No violations"} if you find no violation in the image.
"""


CoT_prompt_list = [_PROMPT_USER_RULE1, _PROMPT_USER_RULE2, _PROMPT_USER_RULE3, _PROMPT_USER_RULE4]

random.seed(1)
def main():

    # Replace 'your_script.py' with the actual name of the Python script you want to execute
    image_folder = "../VLM/10k_images"
    data_split_id = "random1"

    with open("dataset10k_metadata.json", "r") as file:
        data_split = json.load(file)
    test_split = data_split['test_split']

    image_list = test_split
    image_list = sorted(image_list)
    tot = len(image_list)

    predictor = Predictor()

    rule_num = 0
    for prompt in CoT_prompt_list:
        rule_num += 1
        for seed_num in range(1):
            seed_num += 1
            seed_str = str(seed_num)
            
            storage_file = "llava_v15_13B_cot_" + data_split_id + "_" + "vqa" + "_" + "rule" + str(rule_num) + "_" + "seed" + seed_str + ".json"
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
                output = predictor.predict(image=image_path, prompt=prompt)
                
                cleared_reply = ''
                for word in output:
                    cleared_reply += word
                cleared_reply = cleared_reply.replace('\n\n', '')

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
