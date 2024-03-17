import time
from flask import Response
from flask_restx import Namespace, Resource
from werkzeug.datastructures import FileStorage
import json
import pickle
import os
import librosa
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import speech_recognition as sr
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import uuid

from response.voice_identification_response import AdditionalInfo, AnalysisResult, ConfidenceScore, VoiceIdentificationResponse

voice_identification_ns = Namespace("Voice Identification related API", description="Voice Identification related API")

voice_identification_parser = voice_identification_ns.parser()
voice_identification_parser.add_argument('file', location='files', type=FileStorage, required = True)



@voice_identification_ns.route('/')
class VoiceIdentification(Resource):
    mimeType: str = 'application/json'    
    path = os.path.join(os.path.abspath('model')+'/'+'team85-voice-recognition.pkl')
    pickeled_model = pickle.load(open(path, 'rb'))
    Xdb: float = 0.0
    emotion: str = ''

    def extract_features(self, file: FileStorage):
        features: list[float] = []
        
        x, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)    
        stft = np.abs(librosa.stft(x))    
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)    
        mel = np.mean(librosa.feature.melspectrogram(y=x, sr=sample_rate).T,axis=0)    
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)    
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sample_rate).T,axis=0)        
        spect_flatness = np.mean(librosa.feature.spectral_flatness(S=stft).T, axis=0)    
        spect_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sample_rate).T, axis=0)
            
        features.append(np.concatenate((mfccs, chroma, mel, contrast, tonnetz, spect_bandwidth, spect_flatness), axis=0))   

        self.Xdb = np.mean(librosa.amplitude_to_db(librosa.feature.rms(y=x), ref=np.max))     

        return features
    
    def SpeakText(self, file: FileStorage):
        if(file.filename.lower().find('.wav') != -1):
            save_path=os.path.join(os.path.abspath('model')+'/'+uuid.uuid4().hex + '.wav')
            file.stream.seek(0)
            file.save(save_path)
            r = sr.Recognizer()        
            with sr.AudioFile(save_path) as source:
                # listen for the data (load audio to memory)
                audio_data = r.record(source)
                # recognize (convert from speech to text)
                text = r.recognize_google(audio_data)
                self.analyze_emotion_tone(text=text)
            os.remove(save_path)
    
    
    def analyze_emotion_tone(self, text: str):
        key = os.environ['COGNITIVE_TEXT_ANALYTICS_KEY']
        endpoint = os.environ['COGNITIVE_TEXT_ANALYTICS_ENDPOINT']
        credential = AzureKeyCredential(key)
        text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
        responses = text_analytics_client.analyze_sentiment(documents=[text],show_opinion_mining=True, show_stats=True)
        if len(responses) > 0:
            self.emotion = responses[0].sentiment
        else :
            self.emotion = ''
    
    @voice_identification_ns.expect(voice_identification_parser)
    def post(self) -> Response:
        try:
            startTime = time.perf_counter()
            voiceType: str = ''            
            result: VoiceIdentificationResponse = VoiceIdentificationResponse(status='success')
            args = voice_identification_parser.parse_args()
            file: FileStorage = args['file']            
            features = np.array(self.extract_features(file=file))
            ss = StandardScaler()
            x_test = ss.fit_transform(features)
            response = self.pickeled_model.predict_proba(x_test)[0]            
            if(response[1]> response[0]) :
                voiceType = 'human'
            else:
                voiceType = 'ai'
            file.stream.seek(0)
            text = self.SpeakText(file)
            result.analysis = AnalysisResult(detectedVoice= response[1]> response[0], voiceType=voiceType)
            result.confidenceScore = ConfidenceScore(aiProbability= response[0] * 100, humanProbability= response[1] * 100)
            result.additionalInfo = AdditionalInfo(backgroundNoiseLevel=self.Xdb, emotionalTone=self.emotion)
            result.responseTime = time.perf_counter() - startTime
            return Response(json.dumps(result, default=lambda obj: obj.__dict__), mimetype=self.mimeType)
        except Exception as err:
            return Response(json.dumps(err.args, default=lambda obj: obj.__dict__), mimetype=self.mimeType) 
