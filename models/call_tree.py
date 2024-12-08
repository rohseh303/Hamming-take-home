from typing import List, Optional, Dict, Tuple
import uuid
from rich.tree import Tree as RichTree
from rich.console import Console
import graphviz
import openai  # You'll need to add this import at the top

console = Console()

class CallNode:
    def __init__(self, user_responses: List[str], agent_transcript: str, response_category: str = "unknown"):
        self.node_id = str(uuid.uuid4())
        self.user_responses = user_responses
        self.agent_transcript = agent_transcript
        self.response_category = response_category
        self.children: List['CallNode'] = []

    def add_child(self, child: 'CallNode'):
        self.children.append(child)

class CallGraph:
    def __init__(self):
        self.root: Optional[CallNode] = None

    def set_root(self, root: CallNode):
        self.root = root

class DAGNode:
    def __init__(self, user_category: str, agent_transcript: str):
        self.id = str(uuid.uuid4())
        self.user_category = user_category
        self.agent_transcript = agent_transcript
        self.out_edges = []  # List of DAGNode

class ConversationDAG:
    def __init__(self):
        self.nodes: Dict[Tuple[str, str], DAGNode] = {}
        self.root: DAGNode = None

    def get_or_create_node(self, user_category: str, agent_transcript: str) -> DAGNode:
        key = (agent_transcript, user_category)
        if key not in self.nodes:
            node = DAGNode(user_category, agent_transcript)
            self.nodes[key] = node
            if self.root is None:
                self.root = node
            return node
        return self.nodes[key]

    def add_edge(self, from_node: DAGNode, to_node: DAGNode):
        from_node.out_edges.append(to_node)

def build_dag_from_callgraph(call_graph: CallGraph) -> ConversationDAG:
    dag = ConversationDAG()
    
    def dfs(call_node: CallNode):
        # The category should now be stored in the CallNode from the exploration phase
        user_category = call_node.response_category  # We'll need to add this field to CallNode
        agent_transcript = call_node.agent_transcript

        current_dag_node = dag.get_or_create_node(user_category, agent_transcript)

        for child in call_node.children:
            child_dag_node = dfs(child)
            dag.add_edge(current_dag_node, child_dag_node)

        return current_dag_node

    if call_graph.root is not None:
        dfs(call_graph.root)
    return dag

def visualize_dag_as_dot(dag: ConversationDAG, filename: str = "conversation_dag"):
    dot = graphviz.Digraph(comment="Conversation DAG")

    # Add nodes
    for (agent_transcript, user_category), node in dag.nodes.items():
        label = f"Agent: {agent_transcript[:20]}...\nUser Type: {user_category}"
        dot.node(node.id, label=label)

    # Add edges
    for node in dag.nodes.values():
        for child in node.out_edges:
            dot.edge(node.id, child.id)

    dot.render(filename, view=True)

# def build_rich_tree(node: CallNode) -> RichTree:
#     # this needs to have 4o to be able to take in all the paths and then group them together + build an actual DAG
#     # Format node text (shorten transcript if too long)
#     transcript_snippet = (node.agent_transcript[:50] + "...") if len(node.agent_transcript) > 50 else node.agent_transcript
#     node_text = f"[bold green]{node.node_id}[/bold green]\nAgent: {transcript_snippet}\nUser: {node.user_responses}"
    
#     rtree = RichTree(node_text)
#     for child in node.children:
#         child_tree = build_rich_tree(child)
#         rtree.add(child_tree)
#     return rtree

# def print_rich_tree(call_graph: CallGraph):
#     if call_graph.root is None:
#         return
#     rtree = build_rich_tree(call_graph.root)
#     console.clear()  # Clear previous output for a "live update" feel
#     console.print(rtree)