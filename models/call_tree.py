from typing import List, Optional, Dict, Tuple
import uuid
from rich.tree import Tree as RichTree
from rich.console import Console
import graphviz
import openai

console = Console()

class CallNode:
    def __init__(self, decision_point: str = None, response_category: str = "unknown", ):
        self.node_id = str(uuid.uuid4())
        self.decision_point = decision_point
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
    def __init__(self, response_category: str, decision_point: str = None):
        self.id = str(uuid.uuid4())
        self.response_category = response_category
        self.decision_point = decision_point
        self.out_edges = set()

class ConversationDAG:
    def __init__(self):
        self.nodes: Dict[str, DAGNode] = {}
        self.root: DAGNode = None

    def get_or_create_node(self, response_category: str, decision_point: str = None) -> DAGNode:
        key = decision_point if decision_point else f"node_{str(uuid.uuid4())}"
        if key not in self.nodes:
            node = DAGNode(response_category, decision_point)
            self.nodes[key] = node
            if self.root is None:
                self.root = node
            return node
        return self.nodes[key]

    def add_edge(self, from_node: DAGNode, to_node: DAGNode):
        from_node.out_edges.add(to_node)

def build_dag_from_callgraph(call_graph: CallGraph) -> ConversationDAG:
    dag = ConversationDAG()
    
    def dfs(call_node: CallNode):
        current_dag_node = dag.get_or_create_node(call_node.response_category, call_node.decision_point)
        print(f"Created DAG node: {current_dag_node.id} with decision point: {current_dag_node.decision_point}")

        for child in call_node.children:
            child_dag_node = dfs(child)
            dag.add_edge(current_dag_node, child_dag_node)
            print(f"Added edge from {current_dag_node.id} to {child_dag_node.id}")

        return current_dag_node

    if call_graph.root is not None:
        dfs(call_graph.root)
    return dag

def visualize_dag_as_dot(dag: ConversationDAG, filename: str = "conversation_dag"):
    dot = graphviz.Digraph(comment="Conversation DAG")
    dot.attr(rankdir='TB')

    # Add all nodes with simple styling
    for node in dag.nodes.values():
        display_text = node.decision_point if node.decision_point else "Start" if node == dag.root else "No decision point"
        dot.node(node.id, label=display_text)

    # Add "End Call" node
    end_call_id = "end_call"
    dot.node(end_call_id, label="End Call", shape="doublecircle")

    # Add edges with response categories as labels
    leaf_nodes = set()
    for node in dag.nodes.values():
        if not node.out_edges:  # Check if the node has no outgoing edges
            leaf_nodes.add(node)
        for child in node.out_edges:
            dot.edge(node.id, child.id, label=child.response_category)

    # Connect leaf nodes to "End Call" node
    for leaf_node in leaf_nodes:
        dot.edge(leaf_node.id, end_call_id, label="complete")

    dot.render(filename, view=True)

def visualize_call_tree(call_graph: CallGraph, filename: str = "conversation_tree"):
    dot = graphviz.Digraph(comment="Conversation Tree")
    dot.attr(rankdir='TB')

    def add_nodes_and_edges(node: CallNode):
        # Add current node
        display_text = node.decision_point if node.decision_point else "No decision point"
        dot.node(node.node_id, label=display_text)

        # Add edges to all children
        for child in node.children:
            dot.node(child.node_id, label=child.decision_point if child.decision_point else "No decision point")
            dot.edge(node.node_id, child.node_id, label=child.response_category)
            add_nodes_and_edges(child)

    # Add start node and connect to root
    if call_graph.root:
        dot.node('start', 'Start', shape='circle', style='filled', fillcolor='#E8FFE8')
        dot.edge('start', call_graph.root.node_id)
        add_nodes_and_edges(call_graph.root)

    dot.render(filename, view=True)
