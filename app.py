from example import example
import json

from flask import Flask, jsonify, request

app = Flask(__name__)
data_text = ""

@app.route('/')
def mainPage():

    return "<h1> <i> THIS IS A RESTRICTED AREA. <br> <br> YOU ARE NOT WELCOMED HERE!!! </i> </h1>"

@app.route('/json', methods=['GET', 'POST'])
def jsonFile():

    data = request.get_json()

    Index = data.get("Review")

    my_list = []

    for person in data["Review"]:
        my_list.append(person["review"])
    
    # for i in my_list:
    #     print("HELLO")
    #     print(i)
    
    ex = example()

    return str(ex.predict(my_list, data_text))
    

if __name__ == '__main__':
    app.run(debug=True, port=4040)