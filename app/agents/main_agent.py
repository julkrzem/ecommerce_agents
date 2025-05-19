from .context_assesment_agent import ContextAssesmentAgent


class MainAgentSupervisor:
    def __init__(self):
        self.context_assesment_agent = ContextAssesmentAgent()

    def invoke(self, question: str) -> str:
        collected_context = ""
        iteration = 0
        max_iteration = 2

        while iteration < max_iteration:
            iteration += 1
            if self.context_assesment_agent.context_sufficient(collected_context,question):
                return "answer based on context"

        return "max iterrations exceeded"