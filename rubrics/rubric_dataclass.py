from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import json
import markdown # Or a simpler custom markdown converter

@dataclass
class RubricCondition:
    description: str
    points_delta: int
    examples: List[str] = field(default_factory=list)
    explanation: Optional[str] = None # Reasoning for points or usage guidance

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)

    def __str__(self):
        example_str = "\n".join([f"    - Example: {ex}" for ex in self.examples])
        expl_str = f"\n    - Explanation: {self.explanation}" if self.explanation else ""
        return f"- Condition: {self.description} [{self.points_delta:+d} pts]\n{example_str}{expl_str}"

@dataclass
class RubricCriterion:
    name: str
    description: str
    conditions: List[RubricCondition] = field(default_factory=list)

    def add_condition(self, condition: RubricCondition):
        self.conditions.append(condition)

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "conditions": [cond.to_dict() for cond in self.conditions],
        }

    @classmethod
    def from_dict(cls, data: dict):
        conditions = [RubricCondition.from_dict(cond) for cond in data.get("conditions", [])]
        return cls(name=data["name"], description=data["description"], conditions=conditions)

    def __str__(self):
        cond_str = "\n".join([str(cond) for cond in self.conditions])
        return f"### {self.name}\n{self.description}\n\n{cond_str}"


@dataclass
class Rubric:
    topic: str
    evaluation_context: str
    base_score: int = 50 # Default starting point
    overall_instructions: Optional[str] = None
    criteria: List[RubricCriterion] = field(default_factory=list)

    def add_criterion(self, criterion: RubricCriterion):
        self.criteria.append(criterion)

    def to_dict(self):
        return {
            "topic": self.topic,
            "evaluation_context": self.evaluation_context,
            "base_score": self.base_score,
            "overall_instructions": self.overall_instructions,
            "criteria": [crit.to_dict() for crit in self.criteria],
        }

    @classmethod
    def from_dict(cls, data: dict):
        criteria = [RubricCriterion.from_dict(crit) for crit in data.get("criteria", [])]
        return cls(
            topic=data["topic"],
            evaluation_context=data["evaluation_context"],
            base_score=data.get("base_score", 50),
            overall_instructions=data.get("overall_instructions"),
            criteria=criteria,
        )

    def to_markdown(self) -> str:
        md = f"# Rubric for: {self.topic}\n"
        md += f"## Evaluation Context: {self.evaluation_context}\n\n"
        md += f"**Base Score:** {self.base_score}\n\n"
        if self.overall_instructions:
            md += f"**Overall Instructions:**\n{self.overall_instructions}\n\n"

        md += "## Criteria:\n\n"
        for criterion in self.criteria:
             md += str(criterion) + "\n\n" # Use __str__ from RubricCriterion

        # Simple Markdown conversion - consider a more robust library if needed
        # md_content = markdown.markdown(md) # Optional: convert full text if needed elsewhere
        return md

    def save_to_json(self, file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_markdown(self, file_path: str):
         with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_markdown())

# Optional: Define RubricInformationTable if needed, potentially inheriting from StormInformationTable
# from ..storm_wiki.modules.storm_dataclass import StormInformationTable
# class RubricInformationTable(StormInformationTable):
#     # Add any rubric-specific methods or adaptations if necessary
#     pass