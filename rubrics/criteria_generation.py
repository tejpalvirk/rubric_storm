import dspy
from typing import Union, List
from .rubric_dataclass import Rubric, RubricCriterion
# Assuming RubricInformationTable is defined or using StormInformationTable
# from .rubric_dataclass import RubricInformationTable
from ..storm_wiki.modules.storm_dataclass import StormInformationTable # Or use adapted one

class GenerateRubricCriteria(dspy.Signature):
    """
    Analyze the curated information about a topic within a specific evaluation context.
    Identify the key dimensions or criteria essential for evaluating options related to this topic and context.
    Output a list of criteria, each with a concise name and a brief description.
    Format:
    Criterion Name 1: Description 1
    Criterion Name 2: Description 2
    ...
    """
    topic = dspy.InputField(prefix="Topic:")
    evaluation_context = dspy.InputField(prefix="Evaluation Context:")
    curated_info = dspy.InputField(prefix="Curated Information (e.g., conversation summary, key findings):")
    criteria_list = dspy.OutputField(prefix="List of Rubric Criteria:\n", format=str)

class RubricCriteriaGenerationModule(dspy.Module):
    def __init__(self, criteria_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.criteria_gen_lm = criteria_gen_lm
        # Consider ChainOfThought if simple Predict is not enough
        self.generate_criteria = dspy.Predict(GenerateRubricCriteria)

    def parse_criteria(self, criteria_text: str) -> List[RubricCriterion]:
        criteria = []
        # Basic parsing, needs robust error handling
        for line in criteria_text.strip().split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                name = parts[0].strip()
                # Remove potential numbering like "1. "
                if '.' in name:
                    name = name.split('.', 1)[1].strip()
                description = parts[1].strip()
                if name and description:
                    criteria.append(RubricCriterion(name=name, description=description))
        return criteria

    def forward(self, topic: str, evaluation_context: str, information_table: StormInformationTable) -> Rubric:
        # TODO: Format information_table content effectively for the prompt
        # Example: Summarize conversations or extract key points
        curated_info_summary = "Placeholder: Summary of curated information..."

        with dspy.settings.context(lm=self.criteria_gen_lm):
            result = self.generate_criteria(
                topic=topic,
                evaluation_context=evaluation_context,
                curated_info=curated_info_summary
            )

        parsed_criteria = self.parse_criteria(result.criteria_list)
        rubric = Rubric(topic=topic, evaluation_context=evaluation_context)
        rubric.criteria = parsed_criteria
        return rubric