from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
import torch
import json
import random

import re

from os import listdir
from os.path import isfile, join

import numpy as np

from datetime import datetime

# Load pre-trained model and tokenizer
path = "/home/xibok/Documents/models--meta-llama--Llama-3.2-3B-Instruct/snapshots"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
# model = AutoModelForCausalLM.from_pretrained("/Users/bok/.cache/huggingface/hub/models--keeeeenw--MicroLlama/snapshots/6403f6afc9c3a34b877603fab3d525842d353b1c", local_files_only=True)

# Read relevant CSV files

inquiry='''Please generate a short bulleted list with descriptions for each variable provided. \n'''

inquiry_transition=inquiry
base_TRACK_path = './DeID CENTER samples/attachments/20250218_TRACK_random_sample/'
TRACK_CSVs_list = [
'sampled_AIS.csv',
'sampled_Biomarkers.csv',
'sampled_Brainmonitoring.csv',
'sampled_CTMRI.csv',
'sampled_CentralHaemostasis.csv',
'sampled_DailyTIL.csv',
'sampled_FollowUp.csv',
'sampled_Genetics.csv',
'sampled_Hospital.csv',
'sampled_HourlyMeasurements.csv',
'sampled_HourlyValues.csv',
'sampled_Imaging.csv',
'sampled_InjuryHx.csv',
'sampled_LabSampling.csv',
'sampled_Labs.csv',
'sampled_MedHx.csv',
'sampled_Medication.csv',
'sampled_Meds.csv',
'sampled_Outcomes.csv',
'sampled_PriorMeds.csv',
'sampled_Subject.csv',
'sampled_Surgeries.csv',
'sampled_SurgeriesCranial.csv',
'sampled_SurgeriesExtraCranial.csv',
'sampled_TransitionsOfCare.csv',
'sampled_Vitals.csv'
]

class TinyChatBot:
    t = 0.45
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

    def chat(self, user_input, inquiry, max_new_tokens=256, temperature=0.7, top_p=0.9):
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

CV_a = []
CD_a = []

TV_a = []
TD_a = []

with open('/home/xibok/Documents/ModerateLlama/latest_tc_vars_desc/CENTER_Vars_05292025_3.txt') as CV:
 with open('/home/xibok/Documents/ModerateLlama/latest_tc_vars_desc/CENTER_Desc_05292025_3.txt') as CD:
  for line in CV:
   CV_a.append(line.strip())
   CD_a.append(CD.readline().strip())

with open('/home/xibok/Documents/ModerateLlama/latest_tc_vars_desc/TRACK_Vars_06042025_0.out') as TV:
 with open('/home/xibok/Documents/ModerateLlama/latest_tc_vars_desc/TRACK_Desc_06042025_0.out') as TD:
  for line in TV:
   TV_a.append(line.strip())
   TD_a.append(TD.readline().strip())

CV_a = np.array(CV_a)
CD_a = np.array(CD_a)

TV_a = np.array(TV_a)
TD_a = np.array(TD_a)

base_TRACK_path = '/home/xibok/Documents/TRACK EXAMPLE UPLOAD 07162024/DATA/tracktbi csv dump processed/transposed/'
TRACK_CSVs_list = [f for f in listdir(base_TRACK_path) if isfile(join(base_TRACK_path, f))]

base_CENTER_path = '/home/xibok/Documents/DeID CENTER samples/attachments/take2/20250218_TRACK_random_sample/transposed/'
CENTER_CSVs_list = [f for f in listdir(base_CENTER_path) if isfile(join(base_CENTER_path, f))]

TRACK_datas = {}
CENTER_datas = {}

for l in range(len(TRACK_CSVs_list)):
  with open(base_TRACK_path + TRACK_CSVs_list[l]) as tcsv:
    for line in tcsv:
      next_one = line.replace('\\N','')
      p = next_one.split('☺')
      TRACK_datas[TRACK_CSVs_list[l].replace('.csv','').split('_')[0] + '.' + p[0]] = re.sub(r',{2,}',',','"'.join(','.join(next_one.split('☺')[1:10]).split('☻')))

for l in range(len(CENTER_CSVs_list)):
  with open(base_CENTER_path + CENTER_CSVs_list[l]) as tcsv:
    for line in tcsv:
      next_one = line.replace('\\N','')
      p = next_one.split(',')
      print(p[0])
      CENTER_datas[p[0]] = re.sub(r',{2,}',',','"'.join(','.join(next_one.split(',')[1:10]).split('"')))

linenum = 0
skipnum = 1

m = 5
n = 5

TRACK_so_far = []
with open('/home/xibok/Documents/ModerateLlama/2025-07-11-sample-data-transform/TRACKlist.out','r') as t:
 for line in t:
  TRACK_so_far.append(line.strip())

CENTER_so_far = []
with open('/home/xibok/Documents/ModerateLlama/2025-07-11-sample-data-transform/CENTERlist.out','r') as c:
 for line in c:
  CENTER_so_far.append(line.strip())

print(len(TRACK_so_far))
print(len(CENTER_so_far))

print(len(np.unique(TRACK_so_far)))
print(len(np.unique(CENTER_so_far)))

ft_dup = []
with open('/home/xibok/Documents/ModerateLlama/2025-07-11-sample-data-transform/TRACK-matches.txt') as ft_f:
  for TVM_us in ft_f:
    ft_dup.append(TVM_us.strip())

fc_dup = []
with open('/home/xibok/Documents/ModerateLlama/2025-07-11-sample-data-transform/CENTER-matches.txt') as fc_f:
  for CVM_us in fc_f:
    fc_dup.append(CVM_us.strip())

ft_dup_final = []
fc_dup_final = []
common_indices = []

fc_dup = np.array(fc_dup)
ft_dup = np.array(ft_dup)
CENTER_so_far = np.array(CENTER_so_far)
TRACK_so_far = np.array(TRACK_so_far)

print(len(ft_dup))
print(len(fc_dup))
print(len(CENTER_so_far))
print(len(TRACK_so_far))

for di in range(len(TRACK_so_far)):
  CENTER_indices = np.where(fc_dup == CENTER_so_far[di])[0]
  TRACK_indices = np.where(ft_dup == TRACK_so_far[di])[0]
  common_index = list(set(CENTER_indices).intersection(set(TRACK_indices)))
  if len(common_index) > 1:
    print(len(common_index))
    print(CENTER_indices)
    print(TRACK_indices)
    print(common_index)
  for ci in common_index:
    common_indices.append(ci)
print(len(common_indices))

for ei in range(len(ft_dup)):
  if ei not in common_indices:
    ft_dup_final.append(ft_dup[ei])
    fc_dup_final.append(fc_dup[ei])

print(len(fc_dup_final))
print(len(ft_dup_final))

#_ = [ ft_dup.remove(n) for n in TRACK_so_far if n in ft_dup ]
#_ = [ fc_dup.remove(n) for n in CENTER_so_far if n in fc_dup ]

#ft = ft_dup_final[0:47]
#fc = fc_dup_final[0:47]

ft = ft_dup_final[47:]
fc = fc_dup_final[47:]

print(len(ft_dup_final))
print(len(fc_dup_final))

print(ft_dup_final[86])
print(fc_dup_final[86])

print(len(np.unique(ft_dup_final)))
print(len(np.unique(fc_dup_final)))


flipit = False
randomize = False
if randomize:
  p = np.random.permutation(len(ft))
  ft = np.array(ft)[p]
  fc = np.array(fc)[p]
  flipstring = 'randomized'
elif flipit:
  ft.reverse()
  fc.reverse()
  flipstring = 'reversed'
else:
  flipstring = 'regular'

fc_index = 0

max_new_tokens=1024#16384#128000-128000//64-128000//8
chatbot = TinyChatBot(model_path=path)
#with open('/home/xibok/Documents/DeID CENTER samples/attachments/CTBI-dictionary_Variables.csv') as f1:
with open('Bot_data_transform_{0}_{1}_t={2}_multi_{3}_{4}_2.out'.format(n,m,chatbot.t,flipstring,random.random()),'w') as f3:
     for TVM in ft:
        print('\n')
        print(datetime.now())
        print('\n')
        if linenum % skipnum == 0:
          user_input_full = ""
        CVM = fc[fc_index]
        fc_index += 1
        if m == 0:
         inquiry_current = '''There are two SQL databasfes that I am trying to harmonize. Are these in the same format.\n'''
        if m == 1:
         inquiry_current = '''There are two SQL databases that I am trying to harmonize. Please compare both and return instructions on how to harmonize.\n'''
        if m == 2:
         inquiry_current = '''There are two SQL databases that I am trying to harmonize. Please return Python code to transform the data format from TRACK-TBI to CENTER-TBI.\n'''
        if m == 3:
         inquiry_current = '''Please return Python code to transform from Database 1 (TRACK-TBI) to Database 2 (CENTER-TBI)\n'''
        if m == 4:
         inquiry_current = '''Return a Python function for data transformation. Provide code only, no prose.'''
        if m == 5:
         inquiry_current = '''Return a Python function to transform to a common language. Provide code only, no prose.'''
        table1 = TVM.split('.')[0]
        table2 = CVM.split('.')[0]
        field1 = TVM.split('.')[1]
        field2 = CVM.split('.')[1]
        description1 = TD_a[np.where(TV_a == TVM)[0]]
        description2 = CD_a[np.where(CV_a == TVM)[0]]
        data1 = TRACK_datas[TVM]
        data2 = CENTER_datas[CVM]
        if n == 1:
         user_input = f'Database 1, TRACK-TBI, has a table called {table1} and a field called {table1}.{field1}. The field description is: "{description1}" A sample of 20 values from that TRACK-TBI field is in this array: {field1} {data1} Database 2, CENTER-TBI, has a table called {table2} and a field called {table2}.{field2}. The field description is: "{description2}" A sample of 20 values from that CENTER-TBI field is in this array: {field2} {data2}'
        if n == 2:
         user_input = f'Consortium 1, TRACK-TBI, has a table called {table1} and a field called {table1}.{field1}. The field description is: "{description1}" A sample of 20 values from that TRACK-TBI field is in this array: {field1} {data1} Consortium 2, CENTER-TBI, has a table called {table2} and a field called {table2}.{field2}. The field description is: "{description2}" A sample of 20 values from that CENTER-TBI field is in this array: {field2} {data2}'
        if n == 3:
         user_input = f'Database 1 (TRACK-TBI): {table1}.{field1} "{description1}" {data1} Database 2 (CENTER-TBI): {table2}.{field2} "{description2}" {data2}'
        if n == 4:
         user_input = f'Consortium 1 has a table called {table1} and a field called {table1}.{field1}. The field description is: "{description1}" A sample of 20 values from that field is in this array: {field1} {data1} Consortium 2 has a table called {table2} and a field called {table2}.{field2}. The field description is: "{description2}" A sample of 20 values from that field is in this array: {field2} {data2}'
        if n == 5:
         user_input = f'TRACK-TBI has a table called {table1} and a field called {table1}.{field1}. The field description is: "{description1}" A sample of 20 values from that TRACK-TBI field is in this array: {field1} {data1} CENTER-TBI has a table called {table2} and a field called {table2}.{ield2}. The field description is: "{description2}" A sample of 20 values from that CENTER-TBI field is in this array: {field2} {data2}'
        if n == 6:
         user_input = f'TRACK-TBI has a table called {table1} and a field called {table1}.{field1}. The field description is: "{description1}" A sample of 20 values from that field is in this array: {field1} {data1} CENTER-TBI has a table called {table2} and a field called {table2}.{field2}. The field description is: "{description2}" A sample of 20 values from that field is in this array: {field2} {data2}'
        print(user_input)
        user_input_full += user_input + '\n'
        linenum += 1
        if linenum % skipnum == 0:
          print('Running LLM...')
          response = chatbot.chat(inquiry_current + '\n' + user_input_full, inquiry_current, max_new_tokens=2048)
          f3.write(':::\n')
          f3.write(user_input_full)
          f3.write('>>>\n')
          f3.write(response)
          f3.write('\n')
          f3.flush()
