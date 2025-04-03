from typing import List
from ...interface import Information


class BaseCallbackHandler:
    """Base callback handler to manage callbacks from the Co-STORM pipeline."""

    def on_turn_policy_planning_start(self, **kwargs):
        """Run when the turn policy planning begins, before deciding the direction or goal for the next conversation turn."""
        pass

    def on_expert_action_planning_start(self, **kwargs):
        """Run when the expert action planning begins, preparing to determine the actions that each expert should take."""
        pass

    def on_expert_action_planning_end(self, **kwargs):
        """Run when the expert action planning ends, after deciding the actions that each expert should take."""
        pass

    def on_expert_information_collection_start(self, **kwargs):
        """Run when the expert information collection starts, start gathering all necessary data from selected sources."""
        pass

    def on_expert_information_collection_end(self, info: List[Information], **kwargs):
        """Run when the expert information collection ends, after gathering all necessary data from selected sources."""
        pass

    def on_expert_utterance_generation_end(self, **kwargs):
        """Run when the expert utterance generation ends, before creating responses or statements from each expert."""
        pass

    def on_expert_utterance_polishing_start(self, **kwargs):
        """Run when the expert utterance polishing begins, to refine and improve the clarity and coherence of generated content."""
        pass

    def on_mindmap_insert_start(self, **kwargs):
        """Run when the process of inserting new information into the mindmap starts."""
        pass

    def on_mindmap_insert_end(self, **kwargs):
        """Run when the process of inserting new information into the mindmap ends."""
        pass

    def on_mindmap_reorg_start(self, **kwargs):
        """Run when the reorganization of the mindmap begins, to restructure and optimize the flow of information."""
        pass

    def on_expert_list_update_start(self, **kwargs):
        """Run when the expert list update starts, to modify or refresh the list of active experts."""
        pass

    def on_article_generation_start(self, **kwargs):
        """Run when the article generation process begins, to compile and format the final article content."""
        pass

    def on_warmstart_update(self, message, **kwargs):
        """Run when the warm start process has update."""
        pass
