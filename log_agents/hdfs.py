"""HDFS domain knowledge: log line format, the 29 event templates and what each event means.

Templates and their order are identical to pre-processing/preprocess.py, which produced the
training data, so a raw line maps to the same EventId the models were trained on.
"""
import re

# (EventId, match regex, template)
HDFS_TEMPLATES = [
    ("E1",  r"Adding an already existing block", "Adding an already existing block[*]"),
    ("E2",  r"Verification succeeded for", "Verification succeeded for[*]"),
    ("E3",  r"Served block.+to", "Served block[*]to[*]"),
    ("E4",  r"Got exception while serving.+to", "Got exception while serving[*]to[*]"),
    ("E5",  r"Receiving block.+src:.+dest:", "Receiving block[*]src:[*]dest:[*]"),
    ("E6",  r"Received block.+src:.+dest:.+of size", "Received block[*]src:[*]dest:[*]of size[*]"),
    ("E7",  r"writeBlock.+received exception", "writeBlock[*]received exception[*]"),
    ("E8",  r"PacketResponder.+for block.+Interrupted", "PacketResponder[*]for block[*]Interrupted[*]"),
    ("E9",  r"Received block.+of size.+from", "Received block[*]of size[*]from[*]"),
    ("E10", r"PacketResponder.+Exception", "PacketResponder[*]Exception[*]"),
    ("E11", r"PacketResponder.+for block.+terminating", "PacketResponder[*]for block[*]terminating[*]"),
    ("E12", r":Exception writing block.+to mirror", "[*]:Exception writing block[*]to mirror[*]"),
    ("E13", r"Receiving empty packet for block", "Receiving empty packet for block[*]"),
    ("E14", r"Exception in receiveBlock for block", "Exception in receiveBlock for block[*]"),
    ("E15", r"Changing block file offset of block.+from.+to.+meta file offset to",
     "Changing block file offset of block[*]from[*]to[*]meta file offset to[*]"),
    ("E16", r":Transmitted block.+to", "[*]:Transmitted block[*]to[*]"),
    ("E17", r":Failed to transfer.+to.+got", "[*]:Failed to transfer[*]to[*]got[*]"),
    ("E18", r"Starting thread to transfer block.+to", "Starting thread to transfer block[*]to[*]"),
    ("E19", r"Reopen Block", "Reopen Block[*]"),
    ("E20", r"Unexpected error trying to delete block.+BlockInfo not found in volumeMap",
     "Unexpected error trying to delete block[*]BlockInfo not found in volumeMap[*]"),
    ("E21", r"Deleting block.+file", "Deleting block[*]file[*]"),
    ("E22", r"BLOCK\* NameSystem.+allocateBlock:", "BLOCK* NameSystem[*]allocateBlock:[*]"),
    ("E23", r"BLOCK\* NameSystem.+delete:.+is added to invalidSet of",
     "BLOCK* NameSystem[*]delete:[*]is added to invalidSet of[*]"),
    ("E24", r"BLOCK\* Removing block.+from neededReplications as it does not belong to any file",
     "BLOCK* Removing block[*]from neededReplications as it does not belong to any file[*]"),
    ("E25", r"BLOCK\* ask.+to replicate.+to", "BLOCK* ask[*]to replicate[*]to[*]"),
    ("E26", r"BLOCK\* NameSystem.+addStoredBlock: blockMap updated:.+is added to.+size",
     "BLOCK* NameSystem[*]addStoredBlock: blockMap updated:[*]is added to[*]size[*]"),
    ("E27", r"BLOCK\* NameSystem.+addStoredBlock: Redundant addStoredBlock request received for.+on.+size",
     "BLOCK* NameSystem[*]addStoredBlock: Redundant addStoredBlock request received for[*]on[*]size[*]"),
    ("E28", r"BLOCK\* NameSystem.+addStoredBlock: addStoredBlock request received for.+on.+size.+But it does not belong to any file",
     "BLOCK* NameSystem[*]addStoredBlock: addStoredBlock request received for[*]on[*]size[*]But it does not belong to any file[*]"),
    ("E29", r"PendingReplicationMonitor timed out block", "PendingReplicationMonitor timed out block[*]"),
]
COMPILED_TEMPLATES = [(eid, re.compile(p, re.IGNORECASE), t) for eid, p, t in HDFS_TEMPLATES]
TEMPLATE_TEXT = {eid: t for eid, _, t in HDFS_TEMPLATES}

# 081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating
LOG_LINE_RE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>[A-Z]+)\s+(?P<component>[\w.$]+):\s+(?P<content>.+)$"
)
BLOCK_RE = re.compile(r"blk_-?\d+")
IP_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?::\d+)?")


def match_template(content):
    for eid, pattern, tmpl in COMPILED_TEMPLATES:
        if pattern.search(content):
            return eid, tmpl
    return None, None


# ── Event semantics used by the RCA and Insight agents ──────────────────────
# (lifecycle stage, kind normal|warning|error, description)
EVENT_INFO = {
    "E1":  ("namenode", "warning", "NameNode was asked to add a block it already has (duplicate block report)"),
    "E2":  ("verify", "normal", "Block scanner verified the block checksum successfully"),
    "E3":  ("read", "normal", "DataNode served the block to a client (read)"),
    "E4":  ("read", "warning", "DataNode hit an exception while serving the block to a client"),
    "E5":  ("write", "normal", "DataNode started receiving the block in the write pipeline"),
    "E6":  ("replicate", "normal", "DataNode received a replica of the block from another DataNode"),
    "E7":  ("write", "error", "writeBlock received an exception - the write pipeline broke"),
    "E8":  ("write", "error", "PacketResponder thread was interrupted while acknowledging packets"),
    "E9":  ("write", "normal", "DataNode finished receiving the block (size recorded)"),
    "E10": ("write", "error", "PacketResponder raised an exception while acknowledging packets"),
    "E11": ("write", "normal", "PacketResponder terminated normally after the write"),
    "E12": ("write", "error", "Exception writing the block to the next DataNode (mirror) in the pipeline"),
    "E13": ("write", "warning", "An empty packet was received for the block"),
    "E14": ("write", "error", "Exception inside receiveBlock - the block was not fully received"),
    "E15": ("write", "warning", "Block file offset changed (block re-opened / partial rewrite)"),
    "E16": ("replicate", "normal", "Block transmitted to another DataNode for replication"),
    "E17": ("replicate", "error", "Failed to transfer the block to another DataNode"),
    "E18": ("replicate", "normal", "DataNode started a thread to transfer (replicate) the block"),
    "E19": ("write", "warning", "Block was re-opened (append / recovery)"),
    "E20": ("delete", "error", "Tried to delete a block that is not in the DataNode volume map"),
    "E21": ("delete", "normal", "DataNode deleted the block file"),
    "E22": ("namenode", "normal", "NameNode allocated a new block for a file"),
    "E23": ("delete", "normal", "NameNode scheduled the block for deletion (invalidSet)"),
    "E24": ("namenode", "warning", "Block removed from neededReplications because it belongs to no file"),
    "E25": ("replicate", "normal", "NameNode asked a DataNode to replicate the block"),
    "E26": ("namenode", "normal", "NameNode recorded a stored replica of the block (blockMap updated)"),
    "E27": ("namenode", "warning", "Redundant addStoredBlock request - replica reported twice"),
    "E28": ("namenode", "error", "addStoredBlock request for a block that belongs to no file (orphan block)"),
    "E29": ("replicate", "error", "Pending replication timed out for the block"),
}
ERROR_EVENTS = {e for e, (_, kind, _) in EVENT_INFO.items() if kind == "error"}
WARNING_EVENTS = {e for e, (_, kind, _) in EVENT_INFO.items() if kind == "warning"}


def describe(eid):
    return EVENT_INFO.get(eid, ("unknown", "warning", "Unknown event"))[2]


def parse_event_tokens(text):
    """'E5 E22 5 e11' / '[E5,E22]' -> ['E5', 'E22', 'E5', 'E11'] keeping only known events."""
    out = []
    for tok in re.split(r"[\s,\[\]>|;]+|->|→", text.strip()):
        tok = tok.strip().upper()
        if not tok:
            continue
        if tok.isdigit():
            tok = f"E{int(tok)}"
        if tok in EVENT_INFO:
            out.append(tok)
    return out
