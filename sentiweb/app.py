from flask import (
    Flask, render_template, redirect, request, url_for
)
from bert import tokenizer, model, device
from bert import predict
app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html', title='Info')

@app.route('/about')
def about():
    return render_template('about.html', title='About')

@app.route('/analyze')
@app.route('/analyze/<sentence>')
def inputTest(sentence=''):
    if sentence is '':
        result = sentence
    else:
        result = predict(sentence, model, device)
    return render_template(
        'analyze.html',
        sentence=sentence,
        result=result,
        title='Analyze'
    )

@app.route('/calculate', methods=['POST'])
def calculate():
    if request.method == 'POST':
        temp = request.form['sentence']
    else:
        temp = None
    print(f'cal: {temp}')
    return redirect(
        url_for(
            'inputTest',
            sentence=temp,
        )
    )

if __name__ == '__main__':
    app.run()
