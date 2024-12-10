import json
import time
import logging
import threading
from typing import List, Dict, Any
import openai
from config.settings import OPENAI_API_KEY, WEBHOOK_PORT, TEST_PHONE_NUMBER, MAX_DEPTH, CALLER_SYSTEM_PROMPT, SYSTEM_PROMPT
from models.call_tree import CallNode, CallGraph, build_dag_from_callgraph, visualize_dag_as_dot
from web.webhook import get_public_url, app, wait_for_call_completion, webhook
from services.hamming_api import BASE_URL, MEDIA_URL, start_call, get_recording, transcribe_audio, truncate_history_at_decision_point

openai.api_key = OPENAI_API_KEY

# Suppress Flask access logs
# werkzeug_logger = logging.getLogger('werkzeug')
# werkzeug_logger.setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set to keep track of processed decision point identifiers
PREVIOUS_DECISION_POINTS = set()
PREVIOUS_DECISION_POINTS_LOCK = threading.Lock()

def determine_possible_responses(agent_transcript: str) -> List[str]:
    try:
        logger.info("Determining possible responses using GPT-4...")
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant that generates possible customer responses. "
                "Given the agent's transcript, propose a list of 3-5 possible user responses that explore different conversation branches. "
                "The responses should be natural for a customer who just heard the given transcript."
            )},
            {"role": "user", "content": f"Agent said: '{agent_transcript}'. What are possible user follow-up responses?"}
        ]
        
        # Updated to use new OpenAI API format
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # We can try to parse the response_text. We asked for a list.
        # If the model returns a bullet list or numbered list, we can split by lines.
        lines = response_text.split("\n")
        
        # Extract responses (filter empty lines and limit to ~5)
        possible = [line for line in lines if line.strip() and not line.strip().lower().startswith("agent")]
        
        # If no clear formatting, fallback to treating the entire response as one suggestion
        if not possible:
            possible = [response_text]

        # Limit to a max of 5
        possible = possible[:5]

        logger.info(f"Possible responses generated: {possible}")
        return possible

    except Exception as e:
        logger.error(f"Error generating responses with GPT-4: {e}")
        # Fallback responses if OpenAI fails
        return ["Yes", "No", "Can you clarify more?", "What's next?", "I have another question"]
    
def analyze_conversation_and_get_responses(conversation_history: str) -> List[str]:
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

        # Updated to use new OpenAI API format
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": analysis_system_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        response_text = completion.choices[0].message.content.strip()
        logger.debug(f"Raw GPT-4 response:\n{response_text}")

        # Add validation for expected JSON structure
        try:
            # First, verify it's valid JSON
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
            logger.error(f"JSON decoding failed: {jde}")
            logger.error("GPT-4 response was not valid JSON")
            logger.debug(f"Invalid JSON content:\n{response_text}")
            return []

    except Exception as e:
        logger.error(f"Error during GPT-4 analysis: {str(e)}")
        logger.exception("Full traceback:")
        return []

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

def categorize_response(response: str) -> str:
    """
    Categorize a user response using GPT-4.
    """
    try:
        # Updated to use new OpenAI API format
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are analyzing a conversation. Each time you provide alternate user responses, also assign a 'category' label to each response. Responses with similar intent should have the same category label."},
                {"role": "user", "content": response}
            ],
            temperature=0,  # Use 0 for consistent categorization
            max_tokens=20
        )
        return completion.choices[0].message.content.strip().lower()
    except Exception as e:
        logger.error(f"Error categorizing response: {e}")
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
            temperature=0.0  # Use 0 for consistent results
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

def explore_branches(
    phone_number: str,
    conversation_history: str,
    depth: int,
    max_depth: int,
    visited: set,
    webhook_url: str,
) -> CallNode:
    """
    Recursively explores conversation branches based on decision points.
    """
    global PREVIOUS_DECISION_POINTS
    
    # Check for maximum depth
    if depth >= max_depth:
        logger.info(f"Reached maximum depth {max_depth}. Stopping recursion.")
        return CallNode(None, "max_depth_reached")
    
    # Initialize the node with decision point and response category
    decision_point = ""  # Replace with actual decision point if available
    response_category = categorize_response(current_response)
    node = CallNode(decision_point, response_category)

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
                child_node = explore_branches(
                    phone_number, 
                    updated_history, 
                    depth + 1, 
                    max_depth, 
                    visited, 
                    webhook_url
                )
                # Store both the agent's question and user's response
                child_node.decision_point = agent_question
                child_node.response = alt_response
                node.add_child(child_node)

            except Exception as e:
                logger.error(f"Error exploring branch: {e}")
                continue

    return node

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

    # Explore branches based on the baseline conversation
    logger.info("Exploring conversation branches...")
    root_node = explore_branches(TEST_PHONE_NUMBER, baseline_history, depth=0, max_depth=MAX_DEPTH, visited=visited, webhook_url=WEBHOOK_URL)
    call_graph.set_root(root_node)

    # Convert call graph to DAG and visualize
    conversation_dag = build_dag_from_callgraph(call_graph)
    visualize_dag_as_dot(conversation_dag, "conversation_visualization")

if __name__ == "__main__":
    main()