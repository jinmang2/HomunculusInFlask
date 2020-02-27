from flask import Flask, url_for, request
app = Flask(__name__)

@app.route('/synset', methods = ['POST', 'GET'])
def getSynset():
    correct = request.args.get('correct')
    input = request.args.get('input')
    return correct + ', ' + input
    
if __name__ == '__main__':
    app.run(host="127.0.0.1", port="8080")
