import json
import time
import logging
import threading
from typing import List, Dict, Any
import openai
from config.settings import OPENAI_API_KEY, WEBHOOK_PORT, TEST_PHONE_NUMBER, MAX_DEPTH, CALLER_SYSTEM_PROMPT, SYSTEM_PROMPT
from models.call_tree import CallNode, CallGraph, build_dag_from_callgraph, visualize_dag_as_dot, visualize_call_tree
from web.webhook import get_public_url, app, wait_for_call_completion, webhook
from services.hamming_api import BASE_URL, MEDIA_URL, start_call, get_recording, transcribe_audio, truncate_history_at_decision_point
from ssl import SSLError

openai.api_key = OPENAI_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set to keep track of processed decision point identifiers
PREVIOUS_DECISION_POINTS = set()
PREVIOUS_DECISION_POINTS_LOCK = threading.Lock()
    
def analyze_conversation_and_get_responses(conversation_history: str, max_retries: int = 3) -> List[str]:
    try:
        logger.info("Analyzing conversation and determining alternate responses with GPT-4...")
        logger.debug(f"Input conversation history:\n{conversation_history}")

        # Updated system prompt with explicit JSON structure
        analysis_system_prompt = (
            "You are an assistant that analyzes a conversation between a User and an Agent. "
            "Your task is to identify decision points where the Agent asks a question or provides multiple options, "
            "and suggest 3-5 alternative User responses for each decision point to explore different conversation branches."
            "Each decision point should be different and not just a variation of another decision point."
            "\nYou must return a JSON array with this exact structure:\n"
            "[\n"
            "  {\n"
            '    "agent_line": "The exact line where the agent asks a question or presents options",\n'
            '    "original_user_response": "The user\'s original response to this agent line",\n'
            '    "alternates": ["alternate response 1", "alternate response 2", "alternate response 3"]\n'
            "  },\n"
            "  // Additional decision points follow the same structure\n"
            "]\n\n"
            "Conversation History:\n"
            f"{conversation_history}"
        )

        for attempt in range(max_retries):
            try:
                client = openai.OpenAI()
                completion = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": analysis_system_prompt},
                        *([] if attempt == 0 else [{"role": "user", "content": 
                            "The previous response was not valid JSON. Please ensure your response is a valid JSON array "
                            "with the exact structure specified above."}])
                    ],
                    temperature=0.7 - (attempt * 0.2),  # Reduce temperature on retries
                    max_tokens=1000
                )

                response_text = completion.choices[0].message.content.strip()
                logger.debug(f"Raw GPT-4 response (attempt {attempt + 1}):\n{response_text}")

                # Validation logic
                decision_points = json.loads(response_text)
                
                # Validate it's a list
                if not isinstance(decision_points, list):
                    logger.error("GPT-4 response is valid JSON but not a list")
                    logger.debug(f"Received type: {type(decision_points)}")
                    return []

                # Validate each decision point has required fields
                required_fields = {"agent_line", "original_user_response", "alternates"}
                for i, dp in enumerate(decision_points):
                    missing_fields = required_fields - set(dp.keys())
                    if missing_fields:
                        logger.error(f"Decision point {i} missing required fields: {missing_fields}")
                        logger.debug(f"Decision point content: {dp}")
                        return []

                logger.info(f"Successfully parsed {len(decision_points)} decision points")
                logger.debug(f"Parsed decision points: {json.dumps(decision_points, indent=2)}")
                return decision_points

            except json.JSONDecodeError as jde:
                logger.warning(f"JSON decoding failed (attempt {attempt + 1}): {jde}")
                if attempt == max_retries - 1:  # Last attempt
                    logger.error("All retry attempts failed to produce valid JSON")
                    logger.debug(f"Final invalid JSON content:\n{response_text}")
                    return []
                continue  # Try again

    except Exception as e:
        logger.error(f"Error during GPT-4 analysis: {str(e)}")
        logger.exception("Full traceback:")
        return []

def categorize_response(response: str) -> str:
    """
    Simplify a response into a short, clear phrase.
    """
    try:
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "Convert the response into an extremely brief phrase (2-4 words) that captures its essence. "
                    "Use simple, direct language. If it's a question, make it short and clear.\n\n"
                    "Examples:\n"
                    "Input: 'I've been a member of your service for about 3 years now'\n"
                    "Output: Confirm Membership\n\n"
                    "Input: 'My internet has been really slow for the past two days'\n"
                    "Output: Report Slow Internet\n\n"
                    "Input: 'Could you please connect me with a customer service representative?'\n"
                    "Output: Request Agent Transfer"
                )},
                {"role": "user", "content": response}
            ],
            temperature=0,
            max_tokens=20
        )
        return completion.choices[0].message.content.strip().lower()
    except Exception as e:
        logger.error(f"Error simplifying response: {e}")
        return "other"

def is_semantically_similar(new_response: str, visited_responses: set) -> bool:
    """
    Use GPT-4 to determine if a new response is semantically similar to any previously visited responses.
    """
    try:
        # Convert visited responses to a list for better prompt formatting
        visited_list = list(visited_responses)
        
        prompt = (
            "Compare the new response with the list of previous responses and determine if it's semantically similar "
            "to any of them. Return 'true' if similar, 'false' if unique.\n\n"
            f"New response: '{new_response}'\n\n"
            f"Previous responses: {json.dumps(visited_list, indent=2)}"
        )

        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a semantic similarity analyzer. Compare responses and determine if they convey the same intent or meaning, even if worded differently."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        result = completion.choices[0].message.content.strip().lower()
        return 'true' in result
    except Exception as e:
        logger.error(f"Error checking semantic similarity: {e}")
        return False

def is_similar_decision_point(agent_line: str, previous_decision_points: set) -> bool:
    """
    Use GPT-4 to determine if this agent question/prompt is semantically similar
    to any previously encountered decision points.
    """
    try:
        previous_list = list(previous_decision_points)
        
        prompt = (
            "Compare the new agent question/prompt with the list of previous ones and determine if it's semantically similar "
            "to any of them (i.e., asking for the same type of information or presenting similar choices). "
            "Return 'true' if similar, 'false' if unique.\n\n"
            f"New question: '{agent_line}'\n\n"
            f"Previous questions: {json.dumps(previous_list, indent=2)}"
        )

        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a semantic similarity analyzer for conversation decision points. Compare agent questions/prompts to determine if they are asking for the same type of information or presenting similar choices, even if worded differently."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        result = completion.choices[0].message.content.strip().lower()
        return 'true' in result
    except Exception as e:
        logger.error(f"Error checking decision point similarity: {e}")
        return False

def convert_to_declarative(question: str) -> str:
    """
    Convert a question into a declarative statement using GPT-4.
    """
    try:
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "You are a language assistant. Convert the following question into a concise, declarative statement."
                )},
                {"role": "user", "content": question}
            ],
            temperature=0,  # Use 0 for consistent results
            max_tokens=20
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error converting to declarative: {e}")
        return question

def explore_branches(
    phone_number: str,
    conversation_history: str,
    depth: int,
    max_depth: int,
    visited: set,
    webhook_url: str,
    node: CallNode
) -> CallNode:
    """
    Recursively explores conversation branches based on decision points.
    """
    global PREVIOUS_DECISION_POINTS
    
    # Check for maximum depth
    if depth >= max_depth:
        logger.info(f"Reached maximum depth {max_depth}. Stopping recursion.")
        return node
    
    # Identify decision points in the conversation
    decision_points = analyze_conversation_and_get_responses(conversation_history)
    logger.info(f"Decision points found: {len(decision_points)}")
    
    # If no decision points, return node with final agent transcript
    if not decision_points:
        return node

    # Iterate over each decision point
    for dp in decision_points:
        agent_question = dp.get("agent_line", "")
        original_response = dp.get("original_user_response", "")
        alternates = dp.get("alternates", [])
        
        # Check if we've seen a similar decision point before
        with PREVIOUS_DECISION_POINTS_LOCK:
            if is_similar_decision_point(agent_question, PREVIOUS_DECISION_POINTS):
                logger.info(f"Skipping similar decision point: {agent_question[:50]}...")
                continue
            
            PREVIOUS_DECISION_POINTS.add(agent_question)

        # Truncate conversation history up to the decision point
        truncated_history = truncate_history_at_decision_point(conversation_history, dp)
        
        # Explore each alternate response
        for alt_response in alternates:
            new_history = f"{truncated_history}\nUser: {alt_response}"
            
            if is_semantically_similar(alt_response, {h.split('\n')[-1][6:] for h in visited if h}):
                continue

            visited.add(new_history)

            try:
                new_call_id = start_call(phone_number, new_history, webhook_url)
                if not wait_for_call_completion(new_call_id):
                    continue
                
                audio_data = get_recording(new_call_id)
                agent_transcript = transcribe_audio(audio_data)
                updated_history = f"{new_history}\nAgent: {agent_transcript}"

                # Create child node with both question and response
                child_node = CallNode(convert_to_declarative(agent_question), categorize_response(alt_response))
                child_node = explore_branches(
                    phone_number, 
                    updated_history, 
                    depth + 1, 
                    max_depth, 
                    visited, 
                    webhook_url,
                    child_node
                )
                node.add_child(child_node)

            except Exception as e:
                logger.error(f"Error exploring branch: {e}")
                continue

    return node

def run_baseline_conversation(phone_number: str, webhook_url: str) -> str:
    """
    Runs a baseline conversation starting with the system prompt.
    """
    # Start the initial call
    call_id = start_call(phone_number, "", webhook_url)

    # Wait for the call to complete
    if not wait_for_call_completion(call_id):
        logger.error("Baseline call timed out.")
        return

    # Retrieve and transcribe the recording
    audio_data = get_recording(call_id)
    transcript = transcribe_audio(audio_data)

    return transcript

def main():
    server_thread = threading.Thread(
        target=app.run,
        kwargs={"host": "0.0.0.0", "port": WEBHOOK_PORT},
        daemon=True
    )
    server_thread.start()

    WEBHOOK_URL = get_public_url(WEBHOOK_PORT)
    logger.info(f"Using webhook URL: {WEBHOOK_URL}")
    time.sleep(2)

    logger.info("Running baseline conversation...")
    try:
        baseline_history = run_baseline_conversation(TEST_PHONE_NUMBER, WEBHOOK_URL)
    except Exception as e:
        logger.error(f"Failed to run baseline conversation: {e}")
        return
    
    print(f"Baseline history:\n{baseline_history}")

    # Initialize CallGraph
    call_graph = CallGraph()

    # Initialize visited set to keep track of explored conversation paths
    visited = set()
    visited.add(baseline_history)

    # Initialize the root node with baseline decision point and response category
    root_decision_point = ""  # Determine the initial decision point if applicable
    root_response_category = categorize_response(baseline_history)
    root_node = CallNode(root_decision_point, root_response_category)

    # Explore branches based on the baseline conversation
    logger.info("Exploring conversation branches...")
    root_node = explore_branches(
        TEST_PHONE_NUMBER, 
        baseline_history, 
        depth=0, 
        max_depth=MAX_DEPTH, 
        visited=visited, 
        webhook_url=WEBHOOK_URL,
        node=root_node
    )
    call_graph.set_root(root_node)

    # Convert call graph to DAG and visualize
    conversation_dag = build_dag_from_callgraph(call_graph)
    visualize_dag_as_dot(conversation_dag, "conversation_visualization")
    
    # Also visualize as a tree
    visualize_call_tree(call_graph, "conversation_tree")

if __name__ == "__main__":
    main()