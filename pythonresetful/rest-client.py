import requests
import json
payload = {'port_num': '8'}
url=""
r = requests.get(url, params=payload)
d = r.json()
ports=d["ports"]
for k,v in ports.items():
    print(k)
#print(r.url)
#print(r.json())
#print(r.text)
#print(r.json())
#d = json.loads(r.json())
#print(d)
