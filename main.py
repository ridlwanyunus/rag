import json

from flask import Flask, jsonify, make_response, request

import query
import database

from query import question

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/knowledge/renew', methods=['GET'])
def knowledge():
    database.renew()

    data = {
        "message": "success renew private knowledge"
    }
    return make_response(jsonify(data))

@app.route('/question', methods=['POST'])
def question():
    data = json.loads(request.data)
    question = data['question']
    answer = query.question(question=question)

    return make_response(jsonify(answer), 200)


def main():
    app.run(debug=True, host='0.0.0.0', port=8080)



if __name__ == '__main__':
    main()