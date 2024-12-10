Conversation Analysis and Exploration System
This project is designed to analyze and explore conversation branches using OpenAI's GPT-4, Deepgram for transcription, and a custom API for call management. The system identifies decision points in conversations, generates alternative responses, and visualizes the conversation flow as a Directed Acyclic Graph (DAG).
Features
Conversation Analysis: Uses GPT-4 to identify decision points and suggest alternative responses.
Call Management: Initiates and manages calls using a custom API.
Transcription: Utilizes Deepgram for audio transcription.
Visualization: Generates visual representations of conversation flows as DAGs and trees.
Setup Instructions
Prerequisites
Python 3.8 or higher
ngrok for exposing local servers to the internet
Installation

1. Clone the repository:
analysis
Install dependencies:
txt
Set up environment variables:
Create a .env file in the root directory and add the following:
openai_api_key, 
deepgram_api_key, 
hamming_api_token

4. Run ngrok:
Start ngrok to expose your local server:
5001
Note the public URL provided by ngrok, as it will be used for webhook configuration.
Running the Application

1. Start the application:
py
Access the webhook:
Ensure ngrok is running to expose the webhook endpoint. The application will automatically fetch the public URL.

3. Visualize the conversation:
The conversation flow will be visualized and saved as a .dot file, which can be viewed using Graphviz.
Design Flow

1. Initialization:
The application starts by setting up a Flask server to handle webhooks.
Ngrok is used to expose the local server to the internet, allowing external services to send webhook requests.

2. Baseline Conversation:
A baseline conversation is initiated using the run_baseline_conversation function.
The conversation history is recorded and used as the starting point for further exploration.
Conversation Analysis:
The analyze_conversation_and_get_responses function uses GPT-4 to identify decision points in the conversation.
For each decision point, alternative user responses are generated to explore different conversation paths.

4. Branch Exploration:
The explore_branches function recursively explores each conversation branch based on the decision points and alternative responses.
Calls are initiated for each branch, and the responses are transcribed and analyzed.

5. Graph Construction:
The conversation is represented as a CallGraph, which is converted into a ConversationDAG using the build_dag_from_callgraph function.
The DAG is visualized using Graphviz, providing a clear view of the conversation flow and decision points.

6. Visualization:
The conversation DAG is saved as a .dot file and can be viewed using Graphviz tools.
This visualization helps in understanding the conversation structure and identifying key decision points.
Configuration
Settings: Configure API keys, URLs, and other constants in config/settings.py.