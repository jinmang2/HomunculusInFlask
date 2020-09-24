from jst import predict
from jst import embedding, jst_mb_model
from flask import Flask, url_for, request
app = Flask(__name__)

@app.route('/sentiment', methods=['POST', 'GET'])
def get_sentiment():

    # Text 받아오기
    text = request.args.get('text')

    # Text가 없을 시 None
    if type(text) == type(None):
        return "None"

    # 감정 분석
    return predict(text, embedding, jst_mb_model)

if __name__ == '__main__':
    app.run()
