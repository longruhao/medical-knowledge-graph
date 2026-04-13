import requests
import json

url = "http://127.0.0.1:5007/predict/"

# 前面使用的是 request.get_json()，request.post()，需要将 data 转为 json 格式
# header为什么这么写：告诉服务器，我发送的是 json 格式的数据
headers = {
    "Content-Type": "application/json;charset=utf8"
}

data = {'sample': '白百何的处女座是《与青春有关的日子》，合作的演员是佟大为、陈羽凡'}

res = requests.post(url, data=json.dumps(data), headers=headers)
# print(res.text)
result = json.loads(res.text)
print(json.dumps(result, ensure_ascii=False, indent=4))
