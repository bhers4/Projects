from flask import Flask, request
from flask import render_template
import json
import time
from PIL import Image

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

@app.route('/files', methods=['POST'])
def testFiles():
    print("Testfiles")
    if request.method == 'POST':
        print("In Post")
        for item in request.files.items():
            print("Item: ", item)
            print(type(item[1]))
            try:
                pilImg = Image.open(item[1])
                pilImg.show()
            except:
                print("PIL Image open failed")
        return json.dumps({'status':'OK'})
    return json.dumps({'status':'OK'})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
