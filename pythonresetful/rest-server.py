#!flask/bin/python
import six
from flask import Flask, jsonify, abort, request, make_response, url_for
from contextlib import closing
import socket
#from flask_httpauth import HTTPBasicAuth

app = Flask(__name__, static_url_path="")
"""
auth = HTTPBasicAuth()


@auth.get_password
def get_password(username):
    if username == 'miguel':
        return 'python'
    return None


@auth.error_handler
def unauthorized():
    # return 403 instead of 401 to prevent browsers from displaying the default
    # auth dialog
    return make_response(jsonify({'error': 'Unauthorized access'}), 403)
"""


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': 'Bad request'}), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]

def find_free_ports(num):
    def __free_port():
        with closing(socket.socket(socket.AF_INET,
                                   socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            print("socket name: %s" % s.getsockname()[1])
            return s.getsockname()[1]

    port_set={}
    step=0
    while True:
        port = __free_port()
        if port not in port_set:
            port_set[port]=""

        if len(port_set) >= num:
            return port_set

        step+=1
        if step > 100:
            print("can't find avilable port")
            return None

    return None

"""
def make_public_task(task):
    print(task)
    new_task = {}
    for field in task:
        if field == 'id':
            new_task['uri'] = url_for('get_task', task_id=task['id'],
                                      _external=True)
        else:
            new_task[field] = task[field]
    return new_task
"""


@app.route('/todo/api/v1.0/ports', methods=['GET'])
def get_ports():
    return jsonify({'ports': find_free_ports(8)})

"""
@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    return jsonify({'task': make_public_task(task[0])})


@app.route('/todo/api/v1.0/tasks', methods=['POST'])
#@auth.login_required
def create_task():
    if not request.json or 'title' not in request.json:
        abort(400)
    task = {
        'id': tasks[-1]['id'] + 1 if len(tasks) > 0 else 1,
        'title': request.json['title'],
        'description': request.json.get('description', ""),
        'done': False
    }
    tasks.append(task)
    return jsonify({'task': make_public_task(task)}), 201


@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['PUT'])
#@auth.login_required
def update_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    if not request.json:
        abort(400)
    if 'title' in request.json and \
            not isinstance(request.json['title'], six.string_types):
        abort(400)
    if 'description' in request.json and \
            not isinstance(request.json['description'], six.string_types):
        abort(400)
    if 'done' in request.json and type(request.json['done']) is not bool:
        abort(400)
    task[0]['title'] = request.json.get('title', task[0]['title'])
    task[0]['description'] = request.json.get('description',
                                              task[0]['description'])
    task[0]['done'] = request.json.get('done', task[0]['done'])
    return jsonify({'task': make_public_task(task[0])})


@app.route('/todo/api/v1.0/tasks/<int:task_id>', methods=['DELETE'])
#@auth.login_required
def delete_task(task_id):
    task = [task for task in tasks if task['id'] == task_id]
    if len(task) == 0:
        abort(404)
    tasks.remove(task[0])
    return jsonify({'result': True})
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
