class TinyAGI:
    """
    Represents an autonomous agent. 

    Attributes:
        agent: An instance of `TinyGPT`, used to generate the agent's actions.

    """
def __init__(
        self,
        agent: str
        ):
        """
        Constructs a `TinyAGI` instance.

        Args:
            agent: The name of the agent.
        """

        self.agent = TinyAgent(
            model_name=model,
            request_timeout=500,
            verbose=False
        )
