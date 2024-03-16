import time
from flask import Response
from flask_restx import Namespace, Resource, fields
from werkzeug.datastructures import FileStorage
import json
import pickle
import os
import librosa
import numpy as np
from sklearn.discriminant_analysis import StandardScaler

from response.voice_identification_response import AnalysisResult, ConfidenceScore, VoiceIdentificationResponse

voice_identification_ns = Namespace("Voice Identification related API", description="Voice Identification related API")

voice_identification_parser = voice_identification_ns.parser()
voice_identification_parser.add_argument('file', location='files', type=FileStorage, required = True)

def extract_features(file: FileStorage):
    features: list[float] = []
    
    x, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)    
    stft = np.abs(librosa.stft(x))    
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)    
    mel = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T,axis=0)    
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)    
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sample_rate).T,axis=0)
        
    features.append(np.concatenate((mfccs, chroma, mel, contrast, tonnetz), axis=0))
    return features

@voice_identification_ns.route('/')
class VoiceIdentification(Resource):
    mimeType: str = 'application/json'    
    path = os.path.join(os.path.abspath('model')+'/'+'team85-voice-recognition.pkl')
    pickeled_model = pickle.load(open(path, 'rb'))
    
    @voice_identification_ns.expect(voice_identification_parser)
    def post(self) -> Response:
        try:
            startTime = time.perf_counter()
            voiceType: str = ''
            result: VoiceIdentificationResponse = VoiceIdentificationResponse(status='success')
            args = voice_identification_parser.parse_args()
            file: FileStorage = args['file']            
            features = np.array(extract_features(file=file))
            ss = StandardScaler()
            x_test = ss.fit_transform(features)
            response = self.pickeled_model.predict_proba(x_test)[0]            
            if(response[1]> response[0]) :
                voiceType = 'human'
            else:
                voiceType = 'ai'
            result.analysis = AnalysisResult(detectedVoice= response[1]> response[0], voiceType=voiceType)
            result.confidenceScore = ConfidenceScore(aiProbability= response[0] * 100, humanProbability= response[1] * 100)
            result.responseTime = time.perf_counter() - startTime
            return Response(json.dumps(result, default=lambda obj: obj.__dict__), mimetype=self.mimeType)
        except Exception as err:
            return Response(json.dumps(err.args, default=lambda obj: obj.__dict__), mimetype=self.mimeType) 