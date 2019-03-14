from example import example
import json

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def mainPage():

    return "<h1> <i> THIS IS A RESTRICTED AREA. <br> <br> YOU ARE NOT WELCOMED HERE!!! </i> </h1>"

@app.route('/json', methods=['GET', 'POST'])
def jsonFile():

    data = request.get_json()

    Index = data.get("Index")

    my_list = []

    for person in data["Index"]:
        my_list.append(person["name"])
    
    for i in my_list:
        print("HELLO")
        print(i)
    
    ex = example()

    return str(ex.predict(my_list))
    

if __name__ == '__main__':
    app.run(debug=True)