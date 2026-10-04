"""
LogVerse AI Platform — Root Cause Analysis (RCA) Causal Graph Generator
========================================================================
Constructs directed dependency and causal failure propagation graphs
(Component -> Block -> Event Sequence -> Fault Trigger -> Root Cause).
Renders interactive HTML graphs for the desktop product UI.
"""

import json
import networkx as nx

# Knowledge mapping of HDFS event codes to human-readable component operations & root causes
EVENT_CAUSALITY_MAP = {
    "E1":  {"name": "Duplicate Block", "category": "Storage", "severity": "Low", "cause": "Redundant block creation request"},
    "E2":  {"name": "Verification Success", "category": "DataIntegrity", "severity": "Info", "cause": "Normal checksum validation"},
    "E3":  {"name": "Block Served", "category": "NetworkRead", "severity": "Info", "cause": "Normal client block read"},
    "E4":  {"name": "Exception Serving Block", "category": "NetworkRead", "severity": "High", "cause": "Socket timeout or premature client disconnection"},
    "E5":  {"name": "Receiving Block", "category": "DataTransfer", "severity": "Info", "cause": "Data streaming initiation"},
    "E6":  {"name": "Received Block", "category": "DataTransfer", "severity": "Info", "cause": "Data block upload completion"},
    "E7":  {"name": "writeBlock Exception", "category": "Disk/NetworkIO", "severity": "Critical", "cause": "Connection reset by peer or I/O write fault"},
    "E8":  {"name": "PacketResponder Interrupted", "category": "Pipeline", "severity": "High", "cause": "Inter-datanode pipeline disruption"},
    "E9":  {"name": "Received Block from Peer", "category": "Replication", "severity": "Info", "cause": "Replication transfer completed"},
    "E10": {"name": "PacketResponder Exception", "category": "Pipeline", "severity": "Critical", "cause": "Pipeline responder network stack exception"},
    "E11": {"name": "PacketResponder Terminated", "category": "Pipeline", "severity": "High", "cause": "Abnormal pipeline teardown due to downstream failure"},
    "E12": {"name": "Exception Writing to Mirror", "category": "Mirroring", "severity": "Critical", "cause": "Secondary DataNode mirror write failure"},
    "E13": {"name": "Empty Packet Received", "category": "Network", "severity": "Medium", "cause": "Unexpected empty packet on pipeline stream"},
    "E14": {"name": "Exception in receiveBlock", "category": "DataNodeCore", "severity": "Critical", "cause": "DataNode block receiver thread crash"},
    "E17": {"name": "Failed to Transfer Block", "category": "Replication", "severity": "High", "cause": "Node unreachable during block replication"},
    "E20": {"name": "BlockInfo Not Found", "category": "VolumeMap", "severity": "High", "cause": "Metadata mismatch in local volume storage"},
    "E24": {"name": "Orphaned Block Replication", "category": "NameNode", "severity": "Medium", "cause": "Block unlinked from active filesystem namespace"},
    "E29": {"name": "Replication Monitor Timeout", "category": "NameNodeMonitor", "severity": "Critical", "cause": "NameNode block replication timeout expired"}
}


class RCAGraphBuilder:
    """
    Constructs a directed graph representing failure propagation and root causes.
    """

    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph_from_session(self, block_id, parsed_records, ml_analysis):
        """
        Builds a causal network graph for a given log block session.
        """
        self.graph = nx.DiGraph()

        # 1. Root System Node
        system_node = f"HDFS Cluster Session: {block_id}"
        self.graph.add_node(system_node, type="system", level=0, color="#3b82f6")

        # 2. Add Component Nodes
        components = set()
        for rec in parsed_records:
            comp = rec.get("Component", "dfs.Core")
            components.add(comp)
            comp_node = f"Component: {comp}"
            if not self.graph.has_node(comp_node):
                self.graph.add_node(comp_node, type="component", level=1, color="#8b5cf6")
                self.graph.add_edge(system_node, comp_node, label="routes to")

        # 3. Add Event Execution Sequence Nodes
        prev_node = None
        root_causes = []

        for idx, rec in enumerate(parsed_records):
            eid = rec.get("EventId", "E0")
            content = rec.get("Content", "")
            comp = rec.get("Component", "dfs.Core")
            comp_node = f"Component: {comp}"

            meta = EVENT_CAUSALITY_MAP.get(eid, {
                "name": f"Event {eid}",
                "category": "Operation",
                "severity": "Info",
                "cause": "Standard log operation"
            })

            is_error = rec.get("IsError", False) or eid in {"E4", "E7", "E8", "E10", "E11", "E12", "E14", "E17", "E20", "E29"}
            node_id = f"Step {idx+1}: {eid} ({meta['name']})"

            node_color = "#ef4444" if is_error else "#10b981"
            if is_error and len(root_causes) == 0:
                node_color = "#dc2626"  # Highlight Primary Root Cause Node
                root_cause_node = f"ROOT CAUSE: {meta['cause']}"
                self.graph.add_node(root_cause_node, type="root_cause", level=3, color="#ff0055", shape="star")
                root_causes.append(root_cause_node)

            self.graph.add_node(node_id, type="event", level=2, color=node_color, details=content, eid=eid, cause=meta['cause'])
            self.graph.add_edge(comp_node, node_id, label="logs")

            if prev_node:
                self.graph.add_edge(prev_node, node_id, label="triggers next")
            prev_node = node_id

            if is_error and root_causes:
                self.graph.add_edge(node_id, root_causes[-1], label="caused by")

        return self.graph, root_causes

    def export_interactive_html(self, height="500px"):
        """
        Exports the NetworkX graph as a standalone interactive HTML string (Vis.js).
        """
        nodes_data = []
        edges_data = []

        for node, attrs in self.graph.nodes(data=True):
            ntype = attrs.get("type", "event")
            color = attrs.get("color", "#3b82f6")
            shape = "diamond" if ntype == "root_cause" else ("box" if ntype == "component" else "dot")
            size = 25 if ntype == "root_cause" else (20 if ntype == "system" else 15)

            nodes_data.append({
                "id": node,
                "label": node,
                "color": {"background": color, "border": "#ffffff"},
                "shape": shape,
                "size": size,
                "font": {"color": "#ffffff", "size": 12},
                "title": f"Type: {ntype}<br>Cause: {attrs.get('cause', 'N/A')}<br>Details: {attrs.get('details', '')[:100]}"
            })

        for u, v, attrs in self.graph.edges(data=True):
            edges_data.append({
                "from": u,
                "to": v,
                "label": attrs.get("label", ""),
                "arrows": "to",
                "color": {"color": "rgba(255,255,255,0.3)"},
                "font": {"color": "#a1a1aa", "size": 10}
            })

        nodes_json = json.dumps(nodes_data)
        edges_json = json.dumps(edges_data)

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
          <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
          <style>
            body {{ margin: 0; background-color: #090d16; font-family: sans-serif; color: #fff; }}
            #mynetwork {{ width: 100%; height: {height}; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; }}
          </style>
        </head>
        <body>
          <div id="mynetwork"></div>
          <script type="text/javascript">
            var container = document.getElementById('mynetwork');
            var data = {{
              nodes: new vis.DataSet({nodes_json}),
              edges: new vis.DataSet({edges_json})
            }};
            var options = {{
              nodes: {{ borderWidth: 2 }},
              edges: {{ smooth: {{ type: 'cubicBezier', forceDirection: 'horizontal' }} }},
              physics: {{
                hierarchicalRepulsion: {{ nodeDistance: 120 }},
                solver: 'forceAtlas2Based'
              }},
              layout: {{
                hierarchical: {{
                  enabled: true,
                  direction: 'LR',
                  sortMethod: 'directed'
                }}
              }}
            }};
            var network = new vis.Network(container, data, options);
          </script>
        </body>
        </html>
        """
        return html_content
