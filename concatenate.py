import json

with open("data.json", "r") as f:
    data = json.loads(f.read())


res = "\n".join([(f"{d['content']}" if d["content"] else "") for d in data])
data = []

for l in res.splitlines():
    if l.strip() != "":
        data.append(l.replace("\n", " "))

data = "\n".join(data)

with open("input.txt", "w") as f:
    f.write(data)
