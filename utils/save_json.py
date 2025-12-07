import json

def save_json(fname,obj):
         with open(fname, "w") as f:
                json.dump(obj,f)