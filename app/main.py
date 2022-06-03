from flask import Flask, request

from app.utils.utils import GetFreeId, SetFreeId

app = Flask(__name__)


@app.route('/f', methods=["POST", "GET"])
def func():
    return 'Hello World!'


@app.route('/get_free_pk', methods=["POST", "GET"])
def usr():
    if request.method == "GET":
        data = request.json
        pk = GetFreeId()()
        if pk is not None:
            data['email'] = pk
            return data
        return {"error": True, "message": "No free device"}


@app.route('/device_stop', methods=["POST", "GET"])
def stop_cmd():
    if request.method == "POST":
        data = request.json
        pk = data['id']
        SetFreeId(pk).set_free_id()
        return {"email": data['email'], "message": "device successfully disconnected"}
    else:
        return {"error": True, "message": "method error"}
