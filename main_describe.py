from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
import torch
import json

import re

from os import listdir
from os.path import isfile, join

# Load pre-trained model and tokenizer
path = "/home/xibok/Documents/models--meta-llama--Llama-3.2-3B-Instruct/snapshots"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained("/Users/bok/.cache/huggingface/hub/models--keeeeenw--MicroLlama/snapshots/6403f6afc9c3a34b877702fab3d525842d353b1c", local_files_only=True)

# Read relevant CSV files

inquiry='''Please generate a short bulleted list with descriptions for each variable provided. \n'''

inquiry_transition=inquiry
base_TRACK_path = '/home/xibok/Documents/DeID CENTER samples/attachments/take2/20250218_TRACK_random_sample/transposed/'

TRACK_CSVs_list = [f for f in listdir(base_TRACK_path) if isfile(join(base_TRACK_path, f))]

class TinyChatBot:
    def __init__(self, model_path="meta-llama/Llama-3.2-3B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.tokenizer.chat_template = """{% for message in messages %}
        {% if message['role'] == 'user' %}
        <s>[INST] {{ message['content'].strip() }} [/INST]
        {% elif message['role'] == 'assistant' %}
        {{ message['content'].strip() }} </s>
        {% endif %}
        {% endfor %}"""

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True
        )
        self.history = []  # List of {"role": "user"/"assistant", "content": str}

    def format_history(self):
        """Format the history using [INST] ... [/INST] blocks."""
        prompt = ""
        for msg in self.history:
            if msg["role"] == "user":
                prompt += f"<s>[INST] {msg['content']} [/INST]"
            elif msg["role"] == "assistant":
                prompt += f"{msg['content']} </s>"
        return prompt

    def chat(self, user_input, inquiry, max_new_tokens=256, temperature=0.45, top_p=0.9):
        """Send a message and get a response from the model."""
        messages= [
            {
                "role": "system",
                "content": inquiry,
            },
            {"role": "user", "content": user_input},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("[/INST]")[-1].strip().split("</s>")[0].strip()

        self.history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        """Clear the conversation history."""
        self.history = []

max_new_tokens=1024#16384#128000-128000//64-128000//8
chatbot = TinyChatBot(model_path=path)
varnum = 0
lindex = 0
next_one = ""
MODE= 0 # 1 = forward,  0 = reverse
DIRECTION_ARRAY = ['bkwd','fwd']
with open('./DeID CENTER samples/attachments/CTBI-dictionary_Variables.csv') as f1:
    table_header = f1.readline()
    with open(f'Bot_07022025_start_stubs_CENTER_headers_rowmajor_2_t=0.45_2_{DIRECTION_ARRAY[MODE]}.out','w') as f3:
        f3.write(':::\n')
        f3.flush()
        # Chat loop
        step = 0
        not_asked_track_yet = True
        for i in range(1,10000000000):  # limit to 5 exchanges
            # User input
            if i == 0:
                user_input = inquiry
            else:
                csv_string_center = ""#inquiry + table_header + '\n' + f1.readline()
                if csv_string_center == "":
                    inquiry_current = inquiry_transition
                    if next_one == "":
                        f2 = open(base_TRACK_path + TRACK_CSVs_list[step if MODE == 1 else len(TRACK_CSVs_list)-1-step])
                    table_name = TRACK_CSVs_list[step if MODE == 1 else len(TRACK_CSVs_list)-1-step].split('_scrambled')[0]#[len#(TRACK_CSVs_list)-1-step]
                    track = ""
                    for _ in range(lindex,lindex+10):
                        next_one = f2.readline().replace('\\N','')
                        if next_one == "":
                            lindex = 0
                            step += 1
                            break
                        #print(table_name)
                        #table_name + '.' + re.sub(r',{2,}',',','"'.join(','.join(next_one.split(',')[0:10]).split('"'))) + '\n'
                        track += re.sub(r',{2,}',',','"'.join(','.join(next_one.split(',')[0:10]).split('"'))) + '\n'
                        lindex = lindex + 10
                    #step += 1
                    track_amalgam = inquiry_current + track
                    user_input = track_amalgam
                else:
                    inquiry_current = inquiry
                    j = 0
                    input_ids = tokenizer(csv_string_center, return_tensors="pt").input_ids
                    input_length = input_ids.shape[1]
                    f1r = None
                    while (input_length < max_new_tokens) and (not f1r == ""):
                        print('{0} '.format(j))
                        f1r = f1.readline()
                        csv_string_center = csv_string_center + '\n' + f1r
                        input_ids = tokenizer(csv_string_center, return_tensors="pt").input_ids
                        input_length = input_ids.shape[1]
                        j = j + 1
                    user_input = csv_string_center
            print('\n')
            print(user_input)
            print('String assembled, waiting for AI response...')
            #tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
            response = chatbot.chat(user_input, inquiry_current, max_new_tokens=1024)#4096)#128000-128000//64)
            print(f"Bot: {response}")
            f3.write(':::')
            f3.write(user_input)
            f3.write('>>>')
            f3.write(json.dumps(response))
            f3.write('\n')
            f3.flush()
