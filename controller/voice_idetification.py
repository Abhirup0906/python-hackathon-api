from flask import Response
from flask_restx import Namespace, Resource, fields
from werkzeug.datastructures import FileStorage
import json

voice_identification_ns = Namespace("Voice Identification related API", description="Voice Identification related API")

voice_identification_parser = voice_identification_ns.parser()
voice_identification_parser.add_argument('file', location='files', type=FileStorage, required = True)

@voice_identification_ns.route('/')
class VoiceIdentification(Resource):
    
    @voice_identification_ns.expect(voice_identification_parser)
    def post(self) -> Response:
        try:
            return Response('')
        except:
            return Response('') 