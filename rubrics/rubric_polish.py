import dspy
from typing import Union, List
from .rubric_dataclass import Rubric, RubricCriterion, RubricCondition # Assuming these are defined

class ReviewRubricConditions(dspy.Signature):
    """
    Review the conditions for the given criterion within the rubric's context.
    Check for:
    - Clarity and Specificity: Is the condition easy to understand and observe?
    - Overlap/Contradiction: Does it overlap significantly or contradict other conditions (within this criterion or potentially globally)?
    - Example Relevance: Do examples accurately illustrate the condition and point delta? Are they appropriate for the evaluation context?
    - Point Logic: Is the point delta (+/-) reasonable given the condition's impact and context? Is +/- 1 used appropriately?
    Output the revised list of conditions for this criterion in the same format as the input (or indicate if no changes needed).
    Format each condition as:
    - Condition: [Description] [+/-N pts]
        - Example: [Example 1 text]
        - Example: [Example 2 text]
        - Explanation: [Explanation text]
    """
    topic = dspy.InputField()
    evaluation_context = dspy.InputField()
    criterion_name = dspy.InputField()
    criterion_description = dspy.InputField()
    conditions_text = dspy.InputField(desc="Text representation of current conditions for the criterion")
    revised_conditions_text = dspy.OutputField(desc="Text representation of revised conditions")


class AddOverallInstructionsAndFinalize(dspy.Signature):
    """
    Given the full text of a rubric (topic, context, criteria, conditions), write concise overall instructions for the LLM judge using it.
    Instructions should cover: how to apply the base score, how to handle missing information for a condition, how to sum points, and how to interpret the final score in context.
    Perform a final check for global consistency. Output the complete, finalized rubric text including the new instructions.
    """
    topic = dspy.InputField()
    evaluation_context = dspy.InputField()
    base_score = dspy.InputField(format=int)
    full_rubric_text = dspy.InputField(desc="Current complete rubric text (excluding overall instructions)")
    final_rubric_text_with_instructions = dspy.OutputField(desc="Complete rubric text including overall instructions")


class RubricPolishingModule(dspy.Module):
    def __init__(self, rubric_polish_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.rubric_polish_lm = rubric_polish_lm
        self.review_conditions = dspy.ChainOfThought(ReviewRubricConditions) # CoT likely needed
        self.finalize_rubric = dspy.Predict(AddOverallInstructionsAndFinalize)

    def parse_revised_conditions(self, text: str) -> List[RubricCondition]:
        conditions = []
        current_condition = None
        # Extremely basic parser - NEEDS TO BE MADE ROBUST
        for line in text.strip().split('\n'):
            line = line.strip()
            if line.startswith("- Condition:"):
                if current_condition: conditions.append(current_condition)
                # Extract description and points
                match = re.match(r"- Condition:\s*(.*?)\s*\[([+-]?\d+)\s*pts?\]", line, re.IGNORECASE)
                if match:
                    desc, points = match.groups()
                    current_condition = RubricCondition(description=desc.strip(), points_delta=int(points), examples=[], explanation=None)
                else: # Fallback if regex fails
                    current_condition = RubricCondition(description=line[len("- Condition:"):].strip(), points_delta=0) # Placeholder points
            elif line.startswith("- Example:") and current_condition:
                current_condition.examples.append(line[len("- Example:"):].strip())
            elif line.startswith("- Explanation:") and current_condition:
                 current_condition.explanation = line[len("- Explanation:"):].strip()
        if current_condition: conditions.append(current_condition)
        return conditions


    def forward(self, rubric: Rubric) -> Rubric:
         with dspy.settings.context(lm=self.rubric_polish_lm):
            revised_rubric = Rubric(
                topic=rubric.topic,
                evaluation_context=rubric.evaluation_context,
                base_score=rubric.base_score
            )

            for criterion in rubric.criteria:
                conditions_text = "\n".join([str(cond) for cond in criterion.conditions])

                review_result = self.review_conditions(
                    topic=rubric.topic,
                    evaluation_context=rubric.evaluation_context,
                    criterion_name=criterion.name,
                    criterion_description=criterion.description,
                    conditions_text=conditions_text
                )

                revised_conditions = self.parse_revised_conditions(review_result.revised_conditions_text)

                # Create new criterion with revised conditions
                revised_criterion = RubricCriterion(
                    name=criterion.name,
                    description=criterion.description,
                    conditions=revised_conditions if revised_conditions else criterion.conditions # Fallback
                )
                revised_rubric.add_criterion(revised_criterion)

            # Generate final instructions and perform final polish
            # Convert the revised rubric (without instructions yet) to text
            current_rubric_text = "\n\n".join([str(c) for c in revised_rubric.criteria])

            finalize_result = self.finalize_rubric(
                topic=revised_rubric.topic,
                evaluation_context=revised_rubric.evaluation_context,
                base_score=revised_rubric.base_score,
                full_rubric_text=current_rubric_text
            )

            # TODO: Parse the finalized_rubric_text_with_instructions
            # This is complex as it involves parsing the *entire* structure back.
            # For simplicity now, let's just update the overall instructions.
            # A more robust approach would re-parse the whole text or have the LLM output instructions separately.
            try:
                 # Attempt to extract instructions (assuming they are added at the start/end)
                 # This is brittle - needs better structure from LLM or parsing
                 instr_match = re.search(r"\*\*Overall Instructions:\*\*\n(.*?)\n\n## Criteria:",
                                         finalize_result.final_rubric_text_with_instructions, re.DOTALL)
                 if instr_match:
                     revised_rubric.overall_instructions = instr_match.group(1).strip()
            except Exception as e:
                 print(f"Could not parse overall instructions: {e}")
                 revised_rubric.overall_instructions = "Default Instructions: Apply conditions additively starting from base score."


         return revised_rubric