import json

def readlines_txt(txt_file):
    with open(txt_file, "r", encoding="utf8") as f :
        text_list = [ i.replace('\n', '') for i in f.readlines()] 
    return text_list

def read_json(json_file): 
    with open (json_file) as f: 
        data = json.loads(f.read())
    return data