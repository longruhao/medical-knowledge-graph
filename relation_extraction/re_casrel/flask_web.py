import json
from flask import Flask, request
from predict import *

# 创建 flask 应用实例
app = Flask(__name__)

# 导入训练好的模型
model_path = '../save_model/best_f1.pth'
my_model = load_model(model_path)


# 定义服务 请求路径 方式
@app.route('/predict/', methods=["POST", "GET"])
def get_relation():
    # 接收数据
    try:
        # sample = request.form.get('sample')
        data = request.get_json()
        # print(data)
        # 对关系进行提取
        outputs = model2predict(data['sample'], my_model)
        # print(outputs)
        if len(outputs) == 0:
            outputs = {'返回结果': '抱歉小主人，您输入的句子过于刁钻，我会继续努力'}
    except:
        outputs = {'返回结果': '抱歉小主人，您输入的句子过于刁钻，我会继续努力'}
    # 返回 json 字符串，ensure_ascii=False 防止中文乱码
    return json.dumps(outputs, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5007)
