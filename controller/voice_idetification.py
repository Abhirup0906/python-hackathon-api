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

        # if self.Xdb < 0: 
        #     self.Xdb = self.Xdb * -1

        # speech_config = speechsdk.SpeechConfig(subscription='48ba41bd7c7b43338664e5639ca30e05', region='eastus')       

        # audio_config = speechsdk.audio.AudioConfig(stream=PushAudioInputStream)
        # speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        # print("Speak into your microphone.")
        # speech_recognition_result = speech_recognizer.recognize_once_async().get()
            
        # r = sr.Recognizer()        
        # with sr.AudioData(file, sample_rate=sample_rate, sample_width=1) as source:
        #     # listen for the data (load audio to memory)
        #     audio_data = r.record(source)
        #     # recognize (convert from speech to text)
        #     text = r.recognize_google(audio_data)
        #     print(text)
        #     return text

        return features
    
    # def SpeakText(self, audio_file: FileStorage) -> str:
    #     # Initialize the recognizer 
    #     r = sr.Recognizer() 
    #     x, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
    #     # MyText = r.recognize_google(audio_file)
    #     # MyText = MyText.lower()
    #     # open the file
    #     with sr.AudioData(audio_file, sample_rate=sample_rate) as source:
    #         # listen for the data (load audio to memory)
    #         audio_data = r.record(source)
    #         # recognize (convert from speech to text)
    #         text = r.recognize_google(audio_data)
    #         print(text)
    #         return text
    
    
    # def analyze_emotion_tone(text):
    #     key = "d4fab660ad334dbb964214e57082dc54"
    #     endpoint = "https://team85languageservice.cognitiveservices.azure.com/"
    #     credential = AzureKeyCredential(key)
    #     text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
    #     response = text_analytics_client.analyze_sentiment(documents=[text],show_opinion_mining=True, show_stats=True)[0]
    #     return response.sentiment, response.confidence_scores

    
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
            #text = self.SpeakText(file)
            result.analysis = AnalysisResult(detectedVoice= response[1]> response[0], voiceType=voiceType)
            result.confidenceScore = ConfidenceScore(aiProbability= response[0] * 100, humanProbability= response[1] * 100)
            result.additionalInfo = AdditionalInfo(backgroundNoiseLevel=self.Xdb)
            result.responseTime = time.perf_counter() - startTime
            return Response(json.dumps(result, default=lambda obj: obj.__dict__), mimetype=self.mimeType)
        except Exception as err:
            return Response(json.dumps(err.args, default=lambda obj: obj.__dict__), mimetype=self.mimeType) 
