import logging
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
import elevenlabs
import google.generativeai as genai
from elevenlabs import voices, set_api_key
import os
import requests
import json
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables
load_dotenv()

# API Keys
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
LEGALMIND_API_URL = os.getenv("LEGALMIND_API_URL", "http://localhost:8000")

# Configure APIs
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")
set_api_key(ELEVEN_LABS_API_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalMindVoiceAssistant:
    def __init__(self, client):
        self.client = client
        self.last_handled_transcript = None
        self.context = {}
        self.legal_keywords = [
            'contract', 'agreement', 'legal', 'clause', 'risk', 'liability', 
            'termination', 'payment', 'analyze', 'review', 'question'
        ]
    
    def is_legal_query(self, text: str) -> bool:
        """Check if the query is legal-related"""
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in self.legal_keywords)
    
    def call_legalmind_api(self, endpoint: str, data: Dict[Any, Any]) -> Optional[Dict]:
        """Make API call to LegalMind backend"""
        try:
            url = f"{LEGALMIND_API_URL}/{endpoint}"
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            response.raise_for_status()
            
            return response.json()
        except requests.RequestException as e:
            logger.error(f"LegalMind API call failed: {e}")
            return None
    
    def process_legal_query(self, transcript: str) -> str:
        """Process legal-related queries through LegalMind backend"""
        transcript_lower = transcript.lower()
        
        # Check for legal questions
        if any(word in transcript_lower for word in ['what is', 'explain', 'help', 'question']):
            result = self.call_legalmind_api("voice/quick_question", {
                "question": transcript,
                "context": self.context.get('last_analysis', '')
            })
            
            if result:
                return result.get('answer', 'I could not process your legal question.')
            else:
                return "I'm having trouble accessing legal information right now. Please consult with a legal professional."
        
        # Default legal response
        return "I can help with contract analysis and legal questions. You can ask me to analyze contracts, explain legal terms, or answer legal questions. How can I assist you today?"
    
    def generate_general_response(self, transcript: str) -> str:
        """Generate response for non-legal queries using Gemini"""
        try:
            response = model.generate_content([
                {"role": "user", "parts": [{"text": f"Respond briefly to: {transcript}"}]}
            ])
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "I'm a legal assistant. I can help you with contract analysis and legal questions. How can I assist you today?"

    def on_begin(self, client, event: BeginEvent):
        print(f"Legal Assistant Session started: {event.id}")
        
        # Check LegalMind backend status
        try:
            response = requests.get(f"{LEGALMIND_API_URL}/voice/status", timeout=5)
            if response.status_code == 200:
                print("✅ Connected to LegalMind backend")
            else:
                print("⚠️ LegalMind backend connection issues")
        except:
            print("❌ Cannot connect to LegalMind backend")

    def on_turn(self, client, event: TurnEvent):
        transcript = event.transcript
        if not event.end_of_turn:
            print(f"User is speaking: {transcript}", end="\r", flush=True)
            return

        if event.end_of_turn and event.turn_is_formatted:
            if transcript == self.last_handled_transcript:
                return
            self.last_handled_transcript = transcript

            print(f"\nUser: {transcript}")

            # Process the query
            if self.is_legal_query(transcript):
                response_text = self.process_legal_query(transcript)
                print("🏛️ Legal Assistant:", response_text)
            else:
                response_text = self.generate_general_response(transcript)
                print("🤖 Assistant:", response_text)

            # Generate and play audio response
            try:
                audio = elevenlabs.generate(
                    text=response_text,
                    voice="21m00Tcm4TlvDq8ikWAM"
                )
                elevenlabs.play(audio)
            except Exception as e:
                logger.error(f"Audio generation failed: {e}")
                print("Audio playback failed")

        elif event.end_of_turn and not event.turn_is_formatted:
            params = StreamingSessionParameters(format_turns=True)
            client.set_params(params)

    def on_terminated(self, client, event: TerminationEvent):
        print(f"Session terminated: {event.audio_duration_seconds} seconds of audio processed")

    def on_error(self, client, error: StreamingError):
        print(f"Error occurred: {error}")

def main():
    print("🏛️ LegalMind Voice Assistant Starting...")
    
    client = StreamingClient(
        StreamingClientOptions(
            api_key=ASSEMBLYAI_API_KEY,
            api_host="streaming.assemblyai.com",
        )
    )

    assistant = LegalMindVoiceAssistant(client)
    client.on(StreamingEvents.Begin, assistant.on_begin)
    client.on(StreamingEvents.Turn, assistant.on_turn)
    client.on(StreamingEvents.Termination, assistant.on_terminated)
    client.on(StreamingEvents.Error, assistant.on_error)

    client.connect(
        StreamingParameters(
            sample_rate=16000,
            format_turns=True,
        )
    )

    try:
        print("🎤 Voice Assistant Ready - Start speaking...")
        print("💡 Try saying: 'What is liability in contracts?' or 'Explain force majeure'")
        client.stream(
            aai.extras.MicrophoneStream(sample_rate=16000)
        )
    finally:
        client.disconnect(terminate=True)

if __name__ == "__main__":
    main()
