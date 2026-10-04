class AgentError(Exception):
    """Raised by an agent when it cannot complete its task (the Coordinator decides what next)."""


class Agent:
    name = "agent"
    task = ""

    def run(self, ctx):
        """Do the work, publish results on ctx.data, return a one-line summary."""
        raise NotImplementedError

    def validate(self, ctx):
        """Return a list of problems with this agent's output (empty = valid)."""
        return []
