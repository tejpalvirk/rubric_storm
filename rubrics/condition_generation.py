import dspy
from typing import Union, List
from .rubric_dataclass import Rubric, RubricCriterion, RubricCondition
# from .rubric_dataclass import RubricInformationTable
from ..storm_wiki.modules.storm_dataclass import StormInformationTable # Or use adapted one
import re # For parsing points

# --- Define Signatures ---

class BrainstormConditions(dspy.Signature):
    """Given an evaluation criterion for a topic/context, and curated info, brainstorm specific, observable conditions (positive or negative attributes/outcomes) relevant to this criterion."""
    topic = dspy.InputField()
    evaluation_context = dspy.InputField()
    criterion_name = dspy.InputField()
    criterion_desc = dspy.InputField()
    curated_info = dspy.InputField(desc="Relevant excerpts from curated info")
    potential_condition_descs = dspy.OutputField(format=List[str])

class RefineConditionDescription(dspy.Signature):
    """Refine the draft condition description to be clear, specific, and observable for the given criterion and context."""
    topic = dspy.InputField()
    evaluation_context = dspy.InputField()
    criterion_name = dspy.InputField()
    potential_condition_desc = dspy.InputField()
    curated_info = dspy.InputField()
    refined_condition_desc = dspy.OutputField(format=str)

class AssignPointDelta(dspy.Signature):
    """
    Assign points (+/-) to this condition for the given criterion/context.
    Default to +1 for positive conditions and -1 for negative conditions unless the curated info strongly justifies a larger impact.
    Explain the reasoning for the point value based on the condition's significance.
    Output format:
    Points Delta: [+/- points, e.g., -1, +3, -5]
    Reasoning: [Explanation]
    """
    topic = dspy.InputField()
    evaluation_context = dspy.InputField()
    criterion_name = dspy.InputField()
    condition_desc = dspy.InputField()
    curated_info = dspy.InputField()
    # Use ChainOfThought here likely
    points_assignment = dspy.OutputField(desc="Contains 'Points Delta:' and 'Reasoning:'")

class GenerateConditionExamples(dspy.Signature):
    """
    Generate 1-3 concrete examples illustrating this specific condition for the given criterion/context.
    Examples should clearly show the condition being met/not met and reference the type of option being evaluated.
    Ground examples in the curated information where possible, citing sources implicitly if needed. Make clear if an example is hypothetical.
    Indicate the points delta applied in the example, e.g., "[+1 pt]", "[-5 pts]".
    """
    topic = dspy.InputField()
    evaluation_context = dspy.InputField()
    criterion_name = dspy.InputField()
    condition_desc = dspy.InputField()
    points_delta = dspy.InputField(format=int)
    curated_info = dspy.InputField()
    # Use ChainOfThought here likely
    examples = dspy.OutputField(format=List[str])


# --- Define Module ---

class RubricConditionGenerationModule(dspy.Module):
    def __init__(self, condition_gen_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.condition_gen_lm = condition_gen_lm
        # TODO: Potentially use different LMs or Predict vs ChainOfThought based on task complexity
        self.brainstorm = dspy.Predict(BrainstormConditions)
        self.refine_desc = dspy.Predict(RefineConditionDescription)
        self.assign_points = dspy.ChainOfThought(AssignPointDelta) # CoT likely needed
        self.gen_examples = dspy.ChainOfThought(GenerateConditionExamples) # CoT likely needed

    def parse_points_assignment(self, assignment_text: str) -> (int, str):
        points_delta = 0
        reasoning = "N/A"
        try:
            match_points = re.search(r"Points Delta:\s*([+-]?\d+)", assignment_text, re.IGNORECASE)
            if match_points:
                points_delta = int(match_points.group(1))

            match_reason = re.search(r"Reasoning:\s*(.*)", assignment_text, re.IGNORECASE | re.DOTALL)
            if match_reason:
                reasoning = match_reason.group(1).strip()
        except Exception as e:
            print(f"Error parsing points assignment: {e} - Text: {assignment_text}")
        # Fallback logic if parsing fails
        if points_delta == 0:
            if "positive" in assignment_text.lower() or "add" in assignment_text.lower() or "+" in assignment_text:
                 points_delta = 1 # Default positive
            elif "negative" in assignment_text.lower() or "subtract" in assignment_text.lower() or "-" in assignment_text:
                 points_delta = -1 # Default negative

        return points_delta, reasoning


    def forward(self, rubric: Rubric, information_table: StormInformationTable) -> Rubric:
        with dspy.settings.context(lm=self.condition_gen_lm):
            for criterion in rubric.criteria:
                # TODO: Select relevant info snippets for this specific criterion
                relevant_info_summary = "Placeholder: Filtered info for criterion..."

                brainstorm_result = self.brainstorm(
                    topic=rubric.topic,
                    evaluation_context=rubric.evaluation_context,
                    criterion_name=criterion.name,
                    criterion_desc=criterion.description,
                    curated_info=relevant_info_summary
                )

                for potential_desc in brainstorm_result.potential_condition_descs:
                    refine_result = self.refine_desc(
                         topic=rubric.topic,
                         evaluation_context=rubric.evaluation_context,
                         criterion_name=criterion.name,
                         potential_condition_desc=potential_desc,
                         curated_info=relevant_info_summary
                    )
                    refined_desc = refine_result.refined_condition_desc

                    if not refined_desc: continue # Skip if refinement fails

                    points_result = self.assign_points(
                        topic=rubric.topic,
                        evaluation_context=rubric.evaluation_context,
                        criterion_name=criterion.name,
                        condition_desc=refined_desc,
                        curated_info=relevant_info_summary
                    )
                    points_delta, reasoning = self.parse_points_assignment(points_result.points_assignment)

                    examples_result = self.gen_examples(
                        topic=rubric.topic,
                        evaluation_context=rubric.evaluation_context,
                        criterion_name=criterion.name,
                        condition_desc=refined_desc,
                        points_delta=points_delta,
                        curated_info=relevant_info_summary
                    )

                    condition = RubricCondition(
                        description=refined_desc,
                        points_delta=points_delta,
                        explanation=reasoning,
                        examples=examples_result.examples
                    )
                    criterion.add_condition(condition)
        return rubric