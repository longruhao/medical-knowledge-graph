from flask import Flask, render_template, request
from predict import *

app = Flask(__name__)

model_path = '../save_model/last_model.pth'
my_model = load_model(model_path)


def fun(text):
    # 返回 关系抽取 结果
    outputs = model2predict(text, my_model)
    return outputs


@app.route('/', methods=['POST', 'GET'])  # GET 保证正常访问、加载页面；POST 接受用户提交的数据
def index():
    question = None
    result = None
    if request.method == 'POST':
        question = request.form['text']
        result = fun(question)
    # render_template 渲染页面，将模板文件与数据结合，返回给浏览器
    return render_template('index.html', question=question, result=result)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5009)
