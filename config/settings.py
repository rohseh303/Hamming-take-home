import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
API_TOKEN = os.getenv("HAMMING_API_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate API Keys
if not API_TOKEN:
    raise ValueError("Hamming API token not found. Please set HAMMING_API_TOKEN in your environment variables.")
if not DEEPGRAM_API_KEY:
    raise ValueError("Deepgram API key not found. Please set DEEPGRAM_API_KEY in your environment variables.")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment variables.")

# URLs and Constants
BASE_URL = "https://app.hamming.ai/api/rest/exercise"
MEDIA_URL = "https://app.hamming.ai/api/media/exercise"
WEBHOOK_PORT = 5001

# +16508798564 - Auto Dealership for new and used cars
TEST_PHONE_NUMBER = "14153580761"
MAX_DEPTH = 3

# Prompts
CALLER_SYSTEM_PROMPT = (
    "You are given a conversation history between a customer and an AI agent."
    "You are to play the role of the cusotmer in the conversation history and ensure that the store AI agent plays it's role according to the conversation history."
    "After playing your part, wait for the store AI agent to play it's part. Then, end the call."
    "Do not mention that you are an AI or that you are testing the system. Just follow the conversation history."
    "If the conversation history is empty, wait for the store AI agent to start the conversation and end the call."
)

DECISION_POINT_SYSTEM_PROMPT = (
    "You are an assistant that identifies decision points in a conversation between a User and an Agent. "
    "A decision point is where the Agent asks a question or offers multiple options. "
    "For each decision point, provide:\n"
    "- The line of the Agent prompting the decision.\n"
    "- The original User response line right after that Agent prompt.\n"
    "- A list of 3-5 alternate User responses that could have been given at that decision point.\n\n"
    "Return the output in JSON format with a list of decision points. Each decision point should have:\n"
    "{\n"
    '  "agent_line": "<agent line>",\n'
    '  "original_user_response": "<original user response>",\n'
    '  "alternates": ["<alt1>", "<alt2>", ...],\n'
    '  "agent_line_index": <line_index_of_agent>,\n'
    '  "user_line_index": <line_index_of_user>\n'
    "}\n"
    "The line indexes correspond to the order in which lines appear in the conversation (starting from 0)."
)

SYSTEM_PROMPT = (
    "You are a customer calling into a business's voice AI system. "
    "Respond naturally to the agent's prompts as a typical customer would. "
    "Inquire about services, ask for pricing, request appointments or reservations, "
    "seek information about operating hours, or ask any other relevant questions based on the business type. "
    "Do not mention that you are an AI or that you are testing the system. Just behave like a normal customer."
)