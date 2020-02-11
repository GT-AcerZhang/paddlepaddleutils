import requests
import json
payload = {'port_num': '8'}
url="http://yq01-sys-hic-v100-box-a223-0155.yq01.baidu.com:8080/todo/api/v1.0/ports"
#r = requests.get('http://yq01-sys-hic-v100-box-a223-0155.yq01.baidu.com:8080/todo/api/v1.0/ports', params=payload)
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
