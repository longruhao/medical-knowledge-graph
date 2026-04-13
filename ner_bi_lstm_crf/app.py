from flask import Flask, render_template, request
from predict import model2test

# index.html 默认路径项目下的 templates，或者用下面更改路径
# app = Flask(__name__, template_folder='path/to/your/templates')
app = Flask(__name__)


def fun(text):
    # 返回 关系抽取 结果
    out = model2test(text)
    return out


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
