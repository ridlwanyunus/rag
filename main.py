import json

from flask import Flask, jsonify, make_response, request

import query
import database
import retirevalqa

app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/knowledge/renew', methods=['GET'])
def knowledge():
    database.renew()

    data = {
        "message": "success renew private knowledge"
    }
    return make_response(jsonify(data), 200)

@app.route('/question', methods=['POST'])
def question():
    data = json.loads(request.data)
    question = data['question']
    qa, memory = retirevalqa.qa_memory()
    # answer = query.conversational_chat_with_memory(qa=qa,question=question, memory=memory)
    answer = query.conversational_chat_with_react(question=question, memory=memory)
    return make_response(jsonify(answer), 200)

@app.route('/refresh', methods=['GET'])
def refresh():
    retirevalqa.clear_memory()

    return make_response(jsonify("Chat history has been cleared"), 200)

def main():
    app.run(debug=True, host='0.0.0.0', port=8080)



if __name__ == '__main__':
    main()