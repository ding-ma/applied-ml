#%%
import json

d = json.load(open("../imagenet_class_index.json"))


rev = {}
for k,v in d.items():
    rev[v[0]] = [k, v[1]]

json.dump(rev, open("folder_to_class.json", "w"))
# %%
