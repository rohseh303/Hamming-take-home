import requests
from io import BytesIO
from deepgram import DeepgramClient, PrerecordedOptions
from config.settings import API_TOKEN, DEEPGRAM_API_KEY, BASE_URL, MEDIA_URL
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Initialize Deepgram SDK (new v3 syntax)
dg_client = DeepgramClient(DEEPGRAM_API_KEY)

def start_call(phone_number: str, conversation_history: str, webhook_url: str) -> str:
    logger.info(f"Starting call with prompt:\n{conversation_history}")
    prompt = f"""Instructions: You are an AI agent acting as the customer in this call. 
                Follow this conversation exactly, speaking only when it's the customer's turn.
                If the conversation continues after you have finished your part in the script, continue to act as the customer.
                If personal details are requested, provide them as if you were the customer.
                \n\nConversation:\n{conversation_history}.
                """
    headers = {
        "Authorization": API_TOKEN,
        "Content-Type": "application/json"
    }
    data = {
        "phone_number": phone_number,
        "prompt": conversation_history,
        "webhook_url": webhook_url
    }

    resp = requests.post(f"{BASE_URL}/start-call", headers=headers, json=data)
    resp.raise_for_status()

    call_id = resp.json()["id"]
    logger.info(f"Started call with id: {call_id}")
    return call_id

def get_recording(call_id: str) -> bytes:
    headers = {
        "Authorization": API_TOKEN
    }
    resp = requests.get(f"{MEDIA_URL}?id={call_id}", headers=headers)
    resp.raise_for_status()
    return resp.content

def transcribe_audio(audio_data: bytes) -> str:
    try:        
        options = PrerecordedOptions(
            smart_format=True,
            model="nova-2",
            language="en-US",
            punctuate=True,
            filler_words=True
        )

        response = dg_client.listen.rest.v("1").transcribe_file(
            {"buffer": audio_data, "mimetype": "audio/wav"},
            options
        )
        transcription = response.results.channels[0].alternatives[0].transcript
        
        if not transcription:
            logger.warning("No transcription received from Deepgram.")
            return "Error: No transcription available."

        return transcription

    except Exception as e:
        logger.error(f"Error during Deepgram transcription: {e}")
        return "Error: Transcription service encountered an issue."