from flask import Flask
from flask import render_template
import json
import time

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/testAjax', methods=['POST'])
def testButton():
    for i in range(5):
        time.sleep(1)
        print(i)
    return json.dumps({'status': 'OK'})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
