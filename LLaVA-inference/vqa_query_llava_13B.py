import json
import os
from predict_sockeye_13B import Predictor
import random

_PROMPT_SYSTEM = """You are a construction site safety inspector." "You are responsible for viewing the given image and give helpful, and polite answers to your supervisor." "You only answer questions that are asked by the supervisor and in the exact way as requested."""

_PROMPT_USER = """
Please read the image and identify if there are violations of the following four safety rules in the image, do not include violations that do not exist in your answer, assume no violation if the visual information is not enough to make a judgement:

1. Use of basic PPE when on foot at construction sites. Machine operators do not need PPE. (hard hats, properly worn clothes covering shoulders and legs, shoes that can cover toes, high-visibility retroreflective vests at night, face shield or safety glasses when cutting, welding, grinding, or drilling).

2. Use of safety harness when working from a height of three meters and the edges are without any edge protection.

3. Adoption of edge protection or edge warning including guardrails, fences, for underground projects three meters in depth with steep retaining wall and for human to stand.

4. Appearance of worker in the blind spots of the operator and within the operation radius of excavators in operation, or excavators with operators inside.

Return {"0": "No violations"} if you find no violation in the image.

An example of what you should be replying:
{"1": "The worker to the left of the image is not wearing a hard hat", "3": "The edge of the excavation to the right is not protected"}.

"""

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

    # for prompt_id, user_input in prompts.items()[4]:
    for seed_num in range(1):
        seed_num += 1
        seed_str = str(seed_num)
        
        storage_file = "llava_v15_13B_one_shot_" + data_split_id + "_" + "vqa" + "_" + "seed" + seed_str + ".json"
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
            output = predictor.predict(image=image_path, prompt=_PROMPT_USER)
            
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
