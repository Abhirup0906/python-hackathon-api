from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from flask_restx import Api

from controller.voice_idetification import voice_identification_ns

app = Flask(__name__)
CORS(app=app, resources={ r"/api/": { "origins": "*" } })

api = Api(app, doc='/swagger-ui')

@app.route('/')
def hello():
    return "Team Cyber Warrior Hackthon 2024 has been started"

api.add_namespace(voice_identification_ns, '/voice/analyze')

if __name__ == '__main__':
    load_dotenv()
    app.run(debug=False)