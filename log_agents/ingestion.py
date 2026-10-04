"""Log Ingestion: pluggable log sources. Manual upload today, Kubernetes later."""
from dataclasses import dataclass


@dataclass
class LogBatch:
    name: str
    text: str
    origin: str


class LogSource:
    origin = "unknown"

    def read(self) -> LogBatch:
        raise NotImplementedError


class UploadSource(LogSource):
    """A file uploaded through the UI or passed on the command line."""
    origin = "upload"

    def __init__(self, name, data: bytes):
        self.name, self.data = name, data

    def read(self):
        return LogBatch(self.name, self.data.decode("utf-8", errors="replace"), self.origin)


class TextSource(LogSource):
    """Text pasted into the UI (raw log lines or event sequences)."""
    origin = "paste"

    def __init__(self, text, name="pasted-input"):
        self.text, self.name = text, name

    def read(self):
        return LogBatch(self.name, self.text, self.origin)


class KubernetesSource(LogSource):
    """Real-time logs from Kubernetes pods (future work).

    Planned design: stream `kubectl logs -f -n <namespace> <pod> --since=<window>` (or the
    Kubernetes API watch endpoint) into fixed time windows, and hand each window to the
    Coordinator as a LogBatch so the same agent workflow runs continuously.
    """
    origin = "kubernetes"
    available = False

    def __init__(self, namespace="default", pod=None, since="10m"):
        self.namespace, self.pod, self.since = namespace, pod, since

    def read(self):
        raise NotImplementedError("Real-time Kubernetes ingestion is planned future work; "
                                  "upload a log file or paste logs for now.")
