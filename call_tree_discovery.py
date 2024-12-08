import json
import time
import logging
import threading
from typing import List, Dict, Any
import openai
from config.settings import OPENAI_API_KEY, WEBHOOK_PORT, TEST_PHONE_NUMBER, MAX_DEPTH, CALLER_SYSTEM_PROMPT, SYSTEM_PROMPT
from models.call_tree import CallNode, CallGraph, build_dag_from_callgraph, visualize_dag_as_dot
from web.webhook import get_public_url, app, wait_for_call_completion, webhook
from services.hamming_api import BASE_URL, MEDIA_URL, start_call, get_recording, transcribe_audio

openai.api_key = OPENAI_API_KEY

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set to keep track of processed decision point identifiers
PROCESSED_DECISION_POINTS = set()
DECISION_POINTS_LOCK = threading.Lock()

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

def run_baseline_conversation(phone_number: str, webhook_url: str) -> str:
    """
    Runs a baseline conversation starting with the system prompt.
    """
    # Initialize conversation history with system prompt as User instruction
    conversation_history = f"User: {CALLER_SYSTEM_PROMPT}"

    # Start the initial call
    call_id = start_call(phone_number, conversation_history, webhook_url)

    # Wait for the call to complete
    if not wait_for_call_completion(call_id):
        logger.error("Baseline call timed out.")
        return conversation_history

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
        return "other"  # Fallback category


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
    # Extract user responses from history
    user_responses = [line[len("User: "):] for line in conversation_history.split('\n') if line.startswith("User:")]

    # Check for maximum depth
    if depth >= max_depth:
        logger.info(f"Reached maximum depth {max_depth}. Stopping recursion.")
        return CallNode(user_responses, "Max depth reached.", "max_depth_reached")

    # Identify decision points in the conversation
    decision_points = analyze_conversation_and_get_responses(conversation_history)
    logger.info(f"Decision points: {decision_points}")
    
    # If no decision points, return node with final agent transcript
    if not decision_points:
        last_agent_line = "No further agent responses."
        node = CallNode(user_responses, last_agent_line, "end_of_conversation")
        return node

    # Initialize last_agent_line
    last_agent_line = conversation_history.split('\n')[-1] if conversation_history else "No agent response."
    
    # Initialize node with the response category from the last user response
    last_user_response = user_responses[-1] if user_responses else ""
    response_category = categorize_response(last_user_response)
    node = CallNode(user_responses, last_agent_line, response_category)

    # Iterate over each decision point
    for dp in decision_points:
        logger.info(f"Decision point: {dp}")
        agent_line = dp.get("agent_line", "")
        original_user_response = dp.get("original_user_response", "")
        alternates = dp.get("alternates", [])

        # Categorize the response before checking duplicates
        response_category = categorize_response(original_user_response)
        
        # Create a more semantic identifier using the response category
        dp_identifier = f"{response_category}-{agent_line[:50]}"

        with DECISION_POINTS_LOCK:
            if dp_identifier in PROCESSED_DECISION_POINTS:
                logger.info(f"Decision point category {dp_identifier} already processed. Skipping.")
                continue
            else:
                PROCESSED_DECISION_POINTS.add(dp_identifier)

        # Truncate conversation history up to before the original user response
        truncated_history = truncate_history_at_decision_point(conversation_history, dp)
        logger.info(f"Truncated history: {truncated_history}")
        for alt_response in alternates:
            # Create new conversation history with the alternate response
            new_history = f"{truncated_history}\nUser: {alt_response}"

            # Avoid revisiting the same history
            if new_history in visited:
                logger.info("Already visited this conversation path. Skipping to avoid loops.")
                continue

            visited.add(new_history)

            # Start a new call with the new history
            logger.info(f"Exploring alternate response: {alt_response}")
            try:
                new_call_id = start_call(phone_number, new_history, webhook_url)
            except Exception as e:
                logger.error(f"Failed to start call with history:\n{new_history}\nError: {e}")
                child_node = CallNode([alt_response], "Error: Failed to start call.", "error")
                node.add_child(child_node)
                continue

            # Wait for the call to complete
            if not wait_for_call_completion(new_call_id):
                logger.error(f"Call {new_call_id} did not complete successfully")
                child_node = CallNode([alt_response], "Error: Call timeout.", "error")
                node.add_child(child_node)
                continue

            # Get and transcribe the recording
            audio_data = get_recording(new_call_id)
            agent_transcript = transcribe_audio(audio_data)

            # Append the agent's response to the new history
            updated_history = f"{new_history}\nAgent: {agent_transcript}"

            # Recursively explore further branches
            child_node = explore_branches(phone_number, updated_history, depth + 1, max_depth, visited, webhook_url)
            node.add_child(child_node)

    return node

def main():
    server_thread = threading.Thread(
        target=app.run,
        kwargs={"host": "0.0.0.0", "port": WEBHOOK_PORT},
        daemon=True
    )
    server_thread.start()

    # Obtain the public webhook URL via ngrok
    WEBHOOK_URL = get_public_url(WEBHOOK_PORT)
    logger.info(f"Using webhook URL: {WEBHOOK_URL}")

    # Give the server some time to start
    time.sleep(2)

    # Run the baseline conversation to get the initial conversation history
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