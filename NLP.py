# %% [markdown]
# # vision text data NLP 
# Tony 08/12/2023
# 
# Input:
# - raw data from my detect_road module
# - chatGPT generated analysis result
# 
# Output:
# - bbox json of specific video 
# - risk json of specific video

# %%
# Let's read the content of the uploaded file to understand the key frame information
file_path = "raw_data/Info_Video_0194.txt"

with open(file_path, 'r') as file:
    key_frame_data = file.read()

# Displaying the first 1000 characters to get an overview of the content
key_frame_data[:1000]


# %%
# Let's read the content of the LLM generated file and load into JSON
file_path = "LLM_data/Result_Video_0194.txt"

with open(file_path, 'r') as file:
    result_data = file.read()

# Displaying the first 1000 characters to get an overview of the content
result_data[:1000]


# %% [markdown]
# 数据筛选清单处理
# 

# %%
# Given content of the "video_human_selection.txt" file
content = """
b1c9c847-3bda4659     1 10 12  19
b1d10d08-c35503b8     7 10 83 90
b1d22ed6-f1cac061       12 29 38 36 35 52 51 66 67 68  77 78 100 107 104 105 157 153 162 163 166 173 181 186 176 160 194 197 170 186 234 236 250 251  295 304 309 295 324 293 357 368 413 415     
b1dac7f7-6b2e0382      1 8 49
b1f4491b-33824f31       4 36  43 44 77  119 120 169 166 170 167 185 209 235
b2ae0446-4b0bfda7      none
b2be7200-b6f7fe0a       1 2 6 13  22 26 29 28 31 34 39 66 67
b2bee3e1-80c787bd      2 6  12 
b2d22b2f-8302eb61       18  44 46 47  51 55  62 63 
b2d502aa-f0b28e3e      12 16  29 30 33
b4542860-0b880bb4      93  240 270  265  295 314 307
cd09a73f-5f6b9212        1  12
cd17ff29-f393274e         1 2 3 4  25 31  37   47 49 68
cd26264b-22001629       2 8  19 47 24 26  12 13 57 9 43 18  46 45  18 43 45 24 48 96 97 107 109 127 151 153
"""
import json
# Process content and convert to desired JSON structure
video_selection = {}
for line in content.strip().split("\n"):
    parts = line.split()
    video_name = parts[0]
    if parts[1] == "none":
        ids = []
    else:
        ids = [int(pid) for pid in parts[1:]]
    video_selection[video_name] = ids

def save_as_json(json_data, json_file_path):
    """Save data as a JSON file."""
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

save_as_json(video_selection, "human_select.json")


# %%


# Process content and convert to desired JSON structure
video_selection = {}
for line in content.strip().split("\n"):
    parts = line.split()
    video_name = parts[0]
    if parts[1] == "none":
        ids = []
        persons_data = {}
    else:
        ids = [int(pid) for pid in parts[1:]]
        persons_data = {str(pid): {"starting_frame": None, "ending_frame": None} for pid in ids}
    
    video_data = {
        "ids": ids
    }
    video_data.update(persons_data)
    video_selection[video_name] = video_data


save_as_json(video_selection, "human_select_interval.json")


# %%
def is_small_bbox(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    # You can adjust these thresholds as needed
    width_threshold = 5.0
    height_threshold = 5.0

    return width < width_threshold or height < height_threshold


# %%
from collections import defaultdict
import re
# Function to parse the raw json_data and extract the relevant information

# Person 0 is on the road 0, sidewalk 1,his/her bbox is [1148.4463   625.3968  1183.7595   663.28143]
# <V> car 1 is at [ 28.309494 712.38245  192.96992  777.90173 ] <VE> 
def parse_raw_data(json_data):
    frames = defaultdict(lambda: {
        "person": defaultdict(lambda: [None, None, None, None]), # Default bounding box for persons
        "car": defaultdict(lambda: [None, None, None, None])    # Default bounding box for cars
    })
    current_frame = None

    for line in json_data.split("\n"):
        if line.startswith("INFO of"):
            current_frame = line.split(":")[0].split()[-1]

        elif "Person" in line and "bbox" in line:
            person_id, bbox_str = re.match(r"Person (\d+) .* bbox is(\[.*?\])", line).groups()
            bbox_str = bbox_str.strip('[]')
            bbox = [float(value) for value in bbox_str.split()]
            if is_small_bbox(bbox):
                bbox =[ None, None, None, None]

            frames[current_frame]['person'][int(person_id)] = (bbox)


        elif "Person" in line and "detected" in line:
            person_id = re.match(r"Person (\d+) is not", line).group(1)

            frames[current_frame]['person'][int(person_id)] = [None,None,None,None]
        elif "<V> car" in line and "<VE>" in line:
            car_id, bbox_str = re.match(r"<V> car (\d+) is at (\[.*?\]) <VE>", line).groups()
            bbox = [float(value) for value in bbox_str.strip('[]').split()]
            
            if is_small_bbox(bbox):
                bbox =[ None, None, None, None]
            frames[current_frame]['car'][int(car_id)] = bbox


    return frames

# Reading the raw json_data file
file_path = "raw_data/Info_Video_0194.txt"

with open(file_path, 'r') as file:
    raw_data_content = file.read()
# Parsing the raw json_data to extract the required information
parsed_data = parse_raw_data(raw_data_content)
parsed_data


# %%


# %%
import json

def fill_missing_frames(json_file_path):
    json_file_path = "JSON_data/" + json_file_path
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    # Find the maximum frame number
    max_frame = int(max(json_data.keys(), key=lambda x: int(x)))

    # Create an empty frame structure
    empty_frame = {
        "person": {},
        "car": {}
    }

    # Fill in the missing frames with the empty frame structure
    for frame_number in range(max_frame + 1):
        frame_key = str(frame_number).zfill(4)
        if frame_key not in json_data:
            json_data[frame_key] = empty_frame

    # Write the filled json_data back to the JSON file
    with open(json_file_path, 'w') as file:
        json.dump(json_data, file, indent=4)

# Example usage
fill_missing_frames('Bbox_Video_274e.json')


# %%
import json
import os

def save_to_json(parsed_data, filename):
    # Create a directory named "JSON_data" if it doesn't exist
    os.makedirs('JSON_data', exist_ok=True)

    # Define the path
    path = os.path.join('JSON_data', filename)

    # Convert the dictionary to a JSON string and write it to the file
    with open(path, 'w') as file:
        json.dump(parsed_data, file, indent=4)

    print(f"json_data has been saved to {path}")



# Usage example
parsed_data = parse_raw_data(raw_data_content)
filename = 'Bbox_Video_0194.json'
save_to_json(parsed_data, filename)


# %%


# %%


# %%


# %%


# %% [markdown]
# ### raw_data -> bbox
# now I have a folder of the files, named 'raw_data/Info_Video_xxxx.txt'. I want you to write a script to go over all these txt file, and :
# - data = read into str
# - parsed_data = parse_raw_data(data)
# - save_to_json(parsed_data,filename)
# For example:
# # Reading the raw data file
# file_path = "raw_data/Info_Video_0194.txt"
# 
# with open(file_path, 'r') as file:
#     raw_data_content = file.read()
# # Parsing the raw data to extract the required information
# parsed_data = parse_raw_data(raw_data_content)
# filename = 'Bbox_Video_0194.json'
# save_to_json(parsed_data, filename)
# 
# give me a script
# 

# %%

# Directory containing the raw data files
raw_data_dir = "raw_data"

# Iterate through each file in the raw data directory
for filename in os.listdir(raw_data_dir):
    if filename.endswith(".txt"):
        # Construct the full path of the file
        file_path = os.path.join(raw_data_dir, filename)

        # Read the raw data from the file
        with open(file_path, 'r') as file:
            raw_data_content = file.read()

        # Parse the raw data
        parsed_data = parse_raw_data(raw_data_content)

        # Generate the output JSON filename
        json_filename = 'Bbox' + filename.replace("Info", "").replace(".txt", ".json")

        # Save the parsed data to a JSON file
        save_to_json(parsed_data, json_filename)
        
        fill_missing_frames(json_filename)


print("Processing completed!")


# %%



# %%


# %%
import os
def preprocess_text(text):
    # Replace bold markdown with unique markers
    text = text.replace('**', '||') # Using '||' as a unique marker for bold text
    
    # Adding distinct delimiters for each person's information
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        if 'Person' in line and ':' in line:
            processed_lines.append('<PERSON_INFO>') # Start of person's information
        processed_lines.append(line)
        if line.strip().endswith(')'):
            processed_lines.append('</PERSON_INFO>') # End of person's information
            
    processed_text = '\n'.join(processed_lines)
    return processed_text


def store_preprocessed_text(filename, preprocessed_text, folder='prep_data'):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Construct the new filename based on the provided name
    base_name = os.path.basename(filename)
    new_name = f"Prep_{base_name.replace('.txt', '.md')}"
    file_path = os.path.join(folder, new_name)

    # Write the preprocessed text to the new file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(preprocessed_text)
    
    print(f"Preprocessed text saved to: {file_path}")

# Example usage
filename = "Info_Video_0194.txt"
preprocessed_text = preprocess_text(result_data)
store_preprocessed_text(filename, preprocessed_text)


# %%

def parse_text(text, bbox_data):
    # Initialize the result
    result = {}

    # Splitting the text into sections based on headers
    sections = text.split('###')

    # Iterate through the sections
    for section in sections:
        if "Risk Evaluation" in section:
            # Extracting frame number
            frame_number_match = re.search(r"Frame (\d+)", section)

            
            if frame_number_match:
                
                # Extracting the last paragraph as text summary
                text_summary = section.strip().split("\n\n")[-1]
                
                frame_number = frame_number_match.group(1)
                result[f"Frame{frame_number.zfill(4)}"] = {
                    "Time": f"{float(frame_number) /2}s",
                    "Text": text_summary,  # place holder here
                    "Pedestrian": {}
                }
           
                # Extracting risk evaluations for persons
                person_info_blocks = section.split('<PERSON_INFO>')[1:]
                for person_info_block in person_info_blocks:

                    person_info = person_info_block.split('</PERSON_INFO>')[0]
                    # person_match = re.search(r"Person (\d+): ([A-Za-z]+)", person_info)
                    person_match = re.search(r"\|\|Person (\d+):\|\| ([A-Za-z]+)", person_info)

                    if person_match:
                        person, risk_level = person_match.groups()
                        risk_level = risk_level.strip()
                        # print("risk_level",risk_level)
                        # Getting the bounding box data
                        bbox = bbox_data[frame_number.zfill(4)]['person'][str(person)]
                        
                        # Storing the extracted information
                        result[f"Frame{frame_number.zfill(4)}"]["Pedestrian"][person] = (risk_level, bbox)

    return result



# Preprocessing the text
preprocessed_text = preprocess_text(result_data)

# Parsing the text into JSON
json_data = parse_text(preprocessed_text, parsed_data)


json_data

# %%
risk,bbox = json_data["Frame0005"]['Pedestrian']['0']
print(risk)
print(bbox)

# %%
save_to_json(json_data,"risk_evaluation_0194.json")

# %%

def load_box_data(json_filename):
    with open(json_filename, 'r') as file:
        box_data = json.load(file)
    return box_data


# %%
from pathlib import Path

result_data_dir = "LLM_data"

for filename in os.listdir(result_data_dir):
    if filename.endswith(".txt"):

        file_path = os.path.join(result_data_dir, filename)

        json_filename = 'risk_evaluation' + filename.replace("Result_Video", "").replace(".txt", ".json")

        output_path = Path('JSON_data/'+json_filename)
        if output_path.exists():
            print(f"Already handled {output_path}")
            continue

        with open(file_path, 'r') as file:
            result_data = file.read()
            file.close()

        pre_file_path = 'prep_data/' + filename
        preprocessed_text = preprocess_text(result_data)
        store_preprocessed_text(pre_file_path, preprocessed_text)

        bbox_json = 'JSON_data/Bbox_Video' + filename.replace("Result_Video", "").replace(".txt", ".json")
        print(bbox_json)
        # Load the corresponding box_data from the JSON file
        box_data = load_box_data(bbox_json)
        
        json_data = parse_text(preprocessed_text, box_data) # need to read form box json

        save_to_json(json_data, json_filename)

print("Processing completed!")


# %%


# %% [markdown]
# I wrote a code to parse the information from the raw_data, but there are two bug within it:
# 1. program don't show person bbox because the raw data don't have it, I want you to record as [None, None, None, None]
# 2. the key of bbox is string, I want it to be int, because it may be more convinent to search
# 
# Debug now
# 
# 

# %% [markdown]
# 非常好，那么让我们处理另外一个问题
# 我现在需要将另外一个名为：all_video_name.json转化为自然语言，但是我只需要其中每个帧的行人数据

# %%
def read_mot_conf(lines):
    """
    Read the MOT file and return a dictionary mapping frame and id to confidence.
    :param mot_file: Path to the MOT formatted file.
    :return: Nested dictionary with frame and id as keys and confidence as value.
    """
    conf_data = {}
    for line in lines:
        parts = line.strip().split(',')
        frame, obj_id, _, _, _, _, conf, _, _, _ = map(float, parts)
        frame = int(frame)
        obj_id = int(obj_id)
        if frame not in conf_data:
            conf_data[frame] = {}
        conf_data[frame][obj_id] = conf

    return conf_data

# %%
def turn_list_to_str(list_p):

    str_p = ",".join(map(str, list_p))
    return str_p

list_p = [161, 141, 135]

print(turn_list_to_str(list_p))
list_p


# %%
import json

def transform_all_json_to_txt(video_name):
    # Load the JSON file
    json_filename = f'./JSON_data/all_json/all_{video_name}.json'
    with open(json_filename, 'r') as f:
        json_data = json.load(f)
    mot_filename = f'./raw_data/p_mot_list/{video_name}.txt'
    with open(mot_filename, 'r') as f:
        lines = f.readlines()

    conf_data = read_mot_conf(lines)

    output = []

    # Convert to natural language description
    for frame, frame_data in json_data.items():
        people = frame_data.get("person", {})
        output.append(f"### frame {frame}")

        person_ids = list(people.keys())
        if not person_ids:
            output.append("no detected person\n")
            continue
        person_ids_str = [int(pid) for pid in people.keys()]  # Convert keys (person IDs) to integers

        output.append(f"people detected: { turn_list_to_str(person_ids_str)}")

        # For each person, display bbox
        for pid, bbox in people.items():
            rounded_bbox = [round(coord, 1) for coord in bbox]
            output.append(f"{pid} is at {rounded_bbox} ")

        # Dummy logic for left/right of car based on bbox x-axis value (assuming bbox format is [x1, y1, x2, y2])
        right_of_car = [int(pid) for pid, bbox in people.items() if bbox[0] > 960]  # using 960 as video shape threshold
        left_of_car = [int(pid) for pid in person_ids if pid not in right_of_car]

        output.append(f"On the right of our car, {turn_list_to_str(right_of_car)}")
        output.append(f"On the left of our car,{turn_list_to_str(left_of_car)}")

        # Logic for high/low confidence using real confidence data
        high_confidence = [pid for pid in person_ids if float(conf_data[int(frame)][int(pid)]) >= 0.7]
        low_confidence = [pid for pid in person_ids if float(conf_data[int(frame)][int(pid)]) <= 0.3]
       


        output.append(f"high confidence: {','.join(high_confidence) if high_confidence else 'none'}")
        output.append(f"low confidence: {','.join(low_confidence) if low_confidence else 'none'} ")
        output.append("")  # For an empty line between frames

    # Save to an output file
    with open(f'./prep_data/0829_J2T/{video_name}.txt', 'w') as f:
        f.write("\n".join(output))
transform_all_json_to_txt('video_0194')

# %%
unproceessed_list = ['b4542860-0b880bb4','cd09a73f-5f6b9212','cd26264b-22001629']

for name in unproceessed_list:
    transform_all_json_to_txt(name)

# %% [markdown]
# # HANDLE J2T
# 

# %%
from pathlib import Path
import os
input_data_dir = "./raw_data/p_mot_list/"
result_data_dir = "prep_data/0829_J2T/"

for filename in os.listdir(input_data_dir):
    if filename.endswith(".txt"):
        video_name = os.path.basename(filename).split('.')[0]

        output_path = result_data_dir + video_name
        if  os.path.exists(output_path):
            print(f"Already handled {output_path}")
            continue
        transform_all_json_to_txt(video_name)
        print(f"Successfully handled {video_name}")
print("Processing completed!")


# %% [markdown]
# @# LLM result parsing

# %%
import re
import json
sample_llm_text = """
<-batch_28->
### Frame 0196

#### Potential Risks
Person 36 is on the left side of the car and has high confidence. This needs to be monitored carefully.
Person 46 is on the left side of the car and has low confidence.
Person 50 is on the left side of the car and has low confidence.

#### Safety Evaluation
Person 36: Risky (Left side, high confidence)
Person 46: Risky (Left side, low confidence)
Person 50: Risky (Left side, low confidence)

### Frame 0197

#### Potential Risks
Person 36 is on the left side of the car and has high confidence. This needs to be monitored carefully.
Person 46 is on the left side of the car and has low confidence.
Person 50 is on the left side of the car and has low confidence.

#### Safety Evaluation
Person 36: Risky (Left side, high confidence)
Person 46: Risky (Left side, low confidence)
Person 50: Risky (Left side, low confidence)

### Frame 0198

#### Potential Risks
Person 36 is on the left side of the car and has high confidence. This needs to be monitored carefully.
Person 46 is on the left side of the car and has low confidence.
Person 50 is on the left side of the car and has low confidence.

#### Safety Evaluation
Person 36: Risky (Left side, high confidence)
Person 46: Risky (Left side, low confidence)
Person 50: Risky (Left side, low confidence)

### Frame 0199

#### Potential Risks
Person 36 is on the left side of the car and has high confidence. This needs to be monitored carefully.
Person 46 is on the left side of the car and has low confidence.
Person 50 is on the left side of the car and has low confidence.

#### Safety Evaluation
Person 36: Risky (Left side, high confidence)
Person 46: Risky (Left side, low confidence)
Person 50: Risky (Left side, low confidence)
"""

def parse_risky_persons(video_name):

    llm_filename = f'./LLM_data/0829_MOT/llm_{video_name}.txt'
    with open(llm_filename, 'r') as f:
        llm_text = f.read()



    # List to store the risky person IDs for each frame
    risky_frames = []
    
    # Regular expression patterns to extract frame number and risky person IDs
    frame_pattern = r"### Frame (\d+)"
    risky_person_pattern = r"Person (\d+): Risky"
    
    # Current frame number
    current_frame = None

    for line in llm_text.strip().split("\n"):
        frame_match = re.search(frame_pattern, line)
        risky_person_match = re.search(risky_person_pattern, line)

        # If a new frame is found
        if frame_match:
            if current_frame is not None:
                risky_frames.append({"frame": current_frame, "risky_persons": risky_persons})
            
            current_frame = int(frame_match.group(1))
            risky_persons = []
        
        # If a risky person is found
        elif risky_person_match:
            risky_person_id = int(risky_person_match.group(1))
            risky_persons.append(risky_person_id)
            
    # Don't forget to add the last frame's data
    if current_frame is not None:
        risky_frames.append({"frame": current_frame, "risky_persons": risky_persons})

    with open(f'./JSON_data/0829_LLM_J2T/llm_{video_name}.json', 'w') as jf:
        json.dump(risky_frames, jf, indent=4)
    return risky_frames

# Sample usage
video_name = 'b1d10d08-c35503b8'
parsed_data = parse_risky_persons(video_name)
print(len(parsed_data))


# %%

from collections import defaultdict
from collections import Counter

def process_parsed_data(parsed_data, video_name):
    processed_data = {}
    processed_data[video_name] = {"ids": []}
    
    # A temporary dictionary to store start and end frames for each risky person
    temp_dict = defaultdict(lambda: {"starting_frame": float("inf"), "ending_frame": -float("inf")})

    for entry in parsed_data:
        frame = entry['frame']
        risky_persons = entry['risky_persons']
        
        # Update the list of unique risky_persons' ids
        for person in risky_persons:
            if person not in processed_data[video_name]["ids"]:
                processed_data[video_name]["ids"].append(person)
        
        # Update starting and ending frames for each risky person
        for person in risky_persons:
            temp_dict[person]["starting_frame"] = min(temp_dict[person]["starting_frame"], frame)
            temp_dict[person]["ending_frame"] = max(temp_dict[person]["ending_frame"], frame)

    # Update the main dictionary with information from temp_dict
    for person, data in temp_dict.items():
        processed_data[video_name][str(person)] = data
    # with open(f'./JSON_data/0829_LLM_J2T/llm_{video_name}.json', 'w') as jf:
    # with open(f'./llm_{video_name}.json', 'w') as jf:
    #     json.dump(processed_data, jf, indent=4)
    return processed_data

# Sample usage
def merge_processed_data(*dicts):
    """
    Merge multiple dictionaries into a single dictionary.
    """
    master_dict = {}
    print(dicts)
    for d in dicts:
        master_dict.update(d)
    return master_dict

# Sample data
sample_parsed_data = [
    {"frame": 12, "risky_persons": []},
    {"frame": 15, "risky_persons": [3, 4]},
    {"frame": 16, "risky_persons": [3, 4, 5]},
    {"frame": 17, "risky_persons": [3, 4, 5]},
    {"frame": 18, "risky_persons": [3, 4, 5, 6]}
]

# Process data for two videos
video_name1 = "b1d10d08-c35503b8"
video_name2 = "video_0194"
processed_data1 = process_parsed_data(sample_parsed_data, video_name1)
processed_data2 = process_parsed_data(sample_parsed_data, video_name2)

# Merge the processed data
master_dict = merge_processed_data(processed_data1, processed_data2)

# Output the master dictionary
# print(master_dict)
master_dict

# %% [markdown]
# ### batch process

# %%
from pathlib import Path
import os
input_data_dir = "./LLM_data/0830_MOT/" #
result_data_dir = "./JSON_data/0830_LLM_J2T/"

master_dict = {}
for filename in os.listdir(input_data_dir):
    if filename.endswith(".txt"):
        file_path  = input_data_dir + filename
        # with open(file_path, 'r') as f:
        #     if '<-end_processing->' in f.read():
        #         print(f"Ready for output {filename}")
        #     else:
        #         continue
                    

        # video_name = os.path.basename(filename).split('.')[0]
        video_name = os.path.basename(filename).split('.')[0][4:]  # Skip the 'llm_' part

        output_path = result_data_dir + video_name
        if os.path.exists(output_path):
            print(f"Already handled {output_path}")
            continue
        parsed_data = parse_risky_persons(video_name)
        video_dict = process_parsed_data(parsed_data,video_name)
        master_dict.update(video_dict)
        print(f"Successfully handled {video_name}")

with open(f'./llm_select_interval_0830.json', 'w') as jf:
    json.dump(master_dict, jf, indent=4)

print("Processing completed!")
# master_dict

# %%
STOP RUN

# %% [markdown]
# ### Optional : NULL Setting

# %%
def reset_frames_in_master_dict(master_dict):
    for video_name, video_data in master_dict.items():
        for person_id in video_data.get('ids', []):
            video_data[str(person_id)]['starting_frame'] = None
            video_data[str(person_id)]['ending_frame'] = None
    return master_dict

# Sample master_dict
sample_master_dict = {
    'b1d10d08-c35503b8': {
        'ids': [3, 4, 5, 6],
        '3': {'starting_frame': 15, 'ending_frame': 18},
        '4': {'starting_frame': 15, 'ending_frame': 18},
        '5': {'starting_frame': 16, 'ending_frame': 18},
        '6': {'starting_frame': 18, 'ending_frame': 18}
    },
    'video_0194': {
        'ids': [3, 4, 5, 6],
        '3': {'starting_frame': 15, 'ending_frame': 18},
        '4': {'starting_frame': 15, 'ending_frame': 18},
        '5': {'starting_frame': 16, 'ending_frame': 18},
        '6': {'starting_frame': 18, 'ending_frame': 18}
    }
}

# Reset the frames
reset_frames_in_master_dict(master_dict)

# Output the updated master_dict to see if frames are reset

with open(f'./llm_select_null.json', 'w') as jf:
    json.dump(master_dict, jf, indent=4)
master_dict

# %%
for key,item in master_dict.items():
    print(len(item['ids']))

# %%


# %%


# %% [markdown]
# ## 断点续传能力
# 用于chatbot宕机的情况，以下是测试样例证明思想有效性

# %% [markdown]
# I notice there are a potential  problem with your improvement:
# here is part of the `read_frames_into_batch`:
# {{{# Batch frames
#     batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
#     concatenated_batches = ['\n'.join(batch) for batch in batches]
#     return concatenated_batches}}}
# 
# The function read_frames_into_batch reads lines from a file and groups them into frames based on a starting marker '### frame'. Then it groups these frames into batches. The output is a list of concatenated frame batches.
# 
# The batch number is determined by batchsize, so if I changed it from 7 to 5, then the `existing_batches` may get differnt. 
# Think an easy method to handle this without changing read_frames_into_batch
# 
# 
# 
# 
# 

# %%
# All available batches
batches = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]

# Scenario 1: Processing all batches
print("Scenario 1: Processing all batches")
for i, batch in enumerate(batches):
    print(f"Processing batch {i} with data {batch}")

print("\n")

# Scenario 2: Resume from a certain point (batch index 2 in this example)
print("Scenario 2: Resuming from batch 2")
existing_batches = 2

for i, batch in enumerate(batches[existing_batches:], start=existing_batches):
    print(f"Processing batch {i} with data {batch}")


# %%
# Sample read_frames_into_batch function
def read_frames_into_batch(frames, batch_size):
    batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    return batches

# Sample function for analyze_videos_with_llm
def analyze_videos_with_llm(frames, last_successful_frame=None, batch_size=7):
    batches = read_frames_into_batch(frames, batch_size=batch_size)
    
    # Calculate the starting batch based on the last_successful_frame
    existing_batches = 0
    if last_successful_frame is not None:
        existing_batches = (last_successful_frame // batch_size)
    
    # Note the use of start=existing_batches
    for i, batch in enumerate(batches[existing_batches:], start=existing_batches):
        print(f"Analyzing starting from batch {i}, frames: {batch}")

# Test data: 20 frames
frames = list(range(1, 21))
print(frames)
# Let's say processing was successful up to frame 10
last_successful_frame = 10

print("Batch size 7")
analyze_videos_with_llm(frames, last_successful_frame, batch_size=7)
print("---")
print("Batch size 5")
analyze_videos_with_llm(frames, last_successful_frame, batch_size=5)


# %% [markdown]
# ### expected output
# ``` 
# "b1d10d08-c35503b8": {
#         "ids": [
#             3,
#             4,
#             5,
#             6
#         ],
#         "3": {
#             "starting_frame": 15,
#             "ending_frame": 18
#         },
#         "4": {
#             "starting_frame": 15,
#             "ending_frame": 18
#         },
#         "5": {
#             "starting_frame": 16,
#             "ending_frame": 18
#         },
#         "6": {
#             "starting_frame": 18,
#             "ending_frame": 18
#         }
#     },
# ```


