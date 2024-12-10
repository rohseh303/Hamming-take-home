import requests
from io import BytesIO
from deepgram import DeepgramClient, PrerecordedOptions
from config.settings import API_TOKEN, DEEPGRAM_API_KEY, BASE_URL, MEDIA_URL, CALLER_SYSTEM_PROMPT
import logging
import openai
from typing import Dict, Any

# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Initialize Deepgram SDK (new v3 syntax)
dg_client = DeepgramClient(DEEPGRAM_API_KEY)

def start_call(phone_number: str, conversation_history: str, webhook_url: str) -> str:
    # logger.info(f"Starting call with prompt:\n{conversation_history}")
    # prompt = f"""Instructions: You are an AI agent acting as the customer in this call. 
    #             Follow this conversation exactly, speaking only when it's the customer's turn.
    #             If the conversation continues after you have finished your part in the script, continue to act as the customer.
    #             If personal details are requested, provide them as if you were the customer.
    #             \n\nConversation:\n{conversation_history}.
    #             """

    # Initialize conversation history with system prompt as User instruction
    conversation_history = f"User: {CALLER_SYSTEM_PROMPT}\n\nConversation:\n{conversation_history}"

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
    
def truncate_history_at_decision_point(conversation_history: str, dp: Dict[str, Any]) -> str:
    try:
        # Extract the agent's question and user's response from the decision point
        agent_line = dp['agent_line']
        user_response = dp['original_user_response']
        
        # First try exact matching
        qa_pair = f"{agent_line} {user_response}"
        qa_pos = conversation_history.find(qa_pair)
        
        if qa_pos != -1:
            # Cut off the history right after the agent's line
            truncated_history = conversation_history[:qa_pos + len(agent_line)]
            return truncated_history
            
        # If exact match fails, try semantic matching with embeddings
        client = openai.OpenAI()
        dp_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=agent_line
        ).data[0].embedding

        # Split the conversation into segments at each speaker change
        segments = []
        current_segment = ""
        for part in conversation_history.split(". "):
            if any(speaker in part for speaker in ["Hi,", "Hello,", "Thank you", "Yes", "No", "Great", "I'm sorry"]):
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = part
            else:
                current_segment += ". " + part
        if current_segment:
            segments.append(current_segment.strip())

        # Find best matching segment
        best_match_score = 0
        best_match_idx = -1

        for i, segment in enumerate(segments):
            segment_embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=segment
            ).data[0].embedding
            
            similarity = cosine_similarity(dp_embedding, segment_embedding)
            
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_idx = i

        if best_match_idx == -1 or best_match_score < 0.8:  # Adjusted threshold
            logger.warning(f"Could not find matching agent line (best score: {best_match_score})")
            return conversation_history
            
        # Truncate at the best matching segment
        truncated_history = ". ".join(segments[:best_match_idx + 1]) + "."
        return truncated_history

    except Exception as e:
        logger.error(f"Error in matching: {e}")
        return conversation_history
    
def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(a * a for a in v1) ** 0.5
    norm2 = sum(b * b for b in v2) ** 0.5
    return dot_product / (norm1 * norm2)