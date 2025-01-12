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
def __update_memory(
            self,
            action: str,
            observation: str,
            update_summary: bool = True
        ):
        """
        Updates the agent's memory with the last action performed and its observation.
        Optionally, updates the summary of agent's history as well.

        Args:
            action (str): The action performed by the ThinkGPT instance.
            observation (str): The observation made by the ThinkGPT 
                instance after performing the action.
            summary (str): The current summary of the agent's history.
            update_summary (bool, optional): Determines whether to update the summary.
        """

        if len(self.encoding.encode(observation)) > self.max_memory_item_size:
            observation = self.summarizer.chunked_summarize(
                observation, self.max_memory_item_size,
                instruction_hint=OBSERVATION_SUMMARY_HINT
                )

        if "memorize_thoughts" in action:
            new_memory = f"ACTION:\nmemorize_thoughts\nTHOUGHTS:\n{observation}\n"
        else:
            new_memory = f"ACTION:\n{action}\nRESULT:\n{observation}\n"

        if update_summary:
            self.summarized_history = self.summarizer.summarize(
                f"Current summary:\n{self.summarized_history}\nAdd to summary:\n{new_memory}",
                self.max_memory_item_size,
                instruction_hint=HISTORY_SUMMARY_HINT
                )

        self.agent.memorize(new_memory)

    def __get_context(self) -> str:
        """
        Retrieves the context for the agent to think and act upon. 

        Returns:
            str: The agent's context.
        """

        summary = len(self.encoding.encode(self.summarized_history))

        if len(self.criticism) > 0:
            criticism = len(self.encoding.encode(self.criticism))
        else:
            criticism = 0

        action_buffer = "\n".join(
                self.agent.remember(
                limit=32,
                sort_by_order=True,
                max_tokens=self.max_context_size - summary - criticism
            )
        )

        return f"SUMMARY\n{self.summarized_history}\nPREV ACTIONS:"\
            f"\n{action_buffer}\n{self.criticism}"
