
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    detectedVoice: bool
    voiceType: str = ''

class ConfidenceScore(BaseModel):
    aiProbability: float = 0.0
    humanProbability: float = 0.0

class AdditionalInfo(BaseModel):
    emotionalTone: str = ''
    backgroundNoiseLevel: float = 0.0

class VoiceIdentificationResponse(BaseModel): 
    status: str
    analysis: AnalysisResult = None
    confidenceScore: ConfidenceScore = None
    additionalInfo: AdditionalInfo = None
    responseTime: int = 0