from flask import Flask
from flask_cors import CORS
from flask_restx import Api

from controller import voice_idetification_ns

app = Flask(__name__)
CORS(app=app, resources={ r"/api/": { "origins": "*" } })

api = Api(app, doc='/swagger-ui')

@app.route('/')
def hello():
    return "Hello World"

api.add_namespace(voice_idetification_ns, '/team85/api/voice-idetification')

if __name__ == '__main__':
    app.run()