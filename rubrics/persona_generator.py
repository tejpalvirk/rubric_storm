import logging
import re
from typing import Union, List

import dspy
import requests
from bs4 import BeautifulSoup

# --- Helper Function (copied from storm_wiki) ---
def get_wiki_page_title_and_toc(url):
    """Get the main title and table of contents from an url of a Wikipedia page."""
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, "html.parser")

        main_title_tag = soup.find("h1")
        if not main_title_tag:
            return None, None # Or raise error
        main_title = main_title_tag.text.replace("[edit]", "").strip().replace("\xa0", " ")

        toc = ""
        levels = []
        # Sections often excluded from Wikipedia ToCs
        excluded_sections = {
            "Contents", "See also", "Notes", "References",
            "Further reading", "External links", "Bibliography", "Sources"
        }

        # Start processing from h2 to exclude the main title from TOC
        for header in soup.find_all(["h2", "h3", "h4", "h5", "h6"]):
            level = int(header.name[1])
            headline_span = header.find(class_='mw-headline')
            if headline_span:
                 section_title = headline_span.text.replace("[edit]", "").strip().replace("\xa0", " ")
            else: # Fallback if no mw-headline span
                 section_title = header.text.replace("[edit]", "").strip().replace("\xa0", " ")

            if section_title in excluded_sections:
                continue

            # Manage indentation levels
            while levels and level <= levels[-1]:
                levels.pop()
            levels.append(level)

            indentation = "  " * (len(levels) - 1) # Indent based on level depth relative to h2
            toc += f"{indentation}{section_title}\n"

        return main_title, toc.strip()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed for {url}: {e}")
        return None, None
    except Exception as e:
        logging.error(f"Error parsing {url}: {e}")
        return None, None


# --- dspy Signatures ---

class FindRelatedSourcesForEvaluation(dspy.Signature):
    """
    I need to create an evaluation rubric for a specific scenario.
    Identify reliable online sources (like comparison sites, expert review blogs, technical specification databases, user forum discussions, standards documents) that are relevant for evaluating the given topic within the specified context.
    List the URLs of these sources, one per line.
    """
    topic = dspy.InputField(prefix="Topic being evaluated:")
    evaluation_context = dspy.InputField(prefix="Specific evaluation context/goal:")
    related_sources = dspy.OutputField(prefix="List of relevant source URLs:\n", format=str)


class GenerateEvaluatorPersona(dspy.Signature):
    """
    For the given topic and evaluation context, define distinct evaluator personas.
    Each persona should represent a different viewpoint relevant to making a judgment (e.g., technical expert, budget-conscious consumer, usability tester, safety inspector, ethical reviewer, end-user with specific needs).
    Use the provided examples of information sources or evaluation criteria for inspiration.
    For each persona, briefly describe their focus during evaluation.
    Format:
    1. [Persona Role 1]: [Description of evaluation focus 1]
    2. [Persona Role 2]: [Description of evaluation focus 2]
    ...
    """
    topic = dspy.InputField(prefix="Topic being evaluated:")
    evaluation_context = dspy.InputField(prefix="Specific evaluation context/goal:")
    examples = dspy.InputField(prefix="Examples of relevant information sources or criteria for inspiration:\n", format=str)
    personas = dspy.OutputField(prefix="List of Evaluator Personas:\n", format=str)

# --- dspy Module ---

class CreateRubricEvaluatorPersona(dspy.Module):
    """Discover different perspectives for evaluating the topic in the given context."""

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        # Using ChainOfThought for potentially more complex reasoning
        self.find_sources = dspy.ChainOfThought(FindRelatedSourcesForEvaluation)
        self.gen_persona = dspy.ChainOfThought(GenerateEvaluatorPersona)
        self.engine = engine

    def forward(self, topic: str, evaluation_context: str):
        with dspy.settings.context(lm=self.engine):
            # Find relevant source types for inspiration (could also use Wikipedia TOCs if applicable)
            related_sources_result = self.find_sources(topic=topic, evaluation_context=evaluation_context)
            related_sources_text = related_sources_result.related_sources

            # Use found sources or just context description as examples for persona gen
            example_text = f"Relevant Source Types Found:\n{related_sources_text}\n\nEvaluation Goal:\n{evaluation_context}"
            if not related_sources_text or related_sources_text.strip() == 'N/A':
                 example_text = f"Evaluation Goal:\n{evaluation_context}"


            gen_persona_output = self.gen_persona(
                topic=topic,
                evaluation_context=evaluation_context,
                examples=example_text
            ).personas

        personas = []
        # Basic parsing, might need improvement
        for line in gen_persona_output.strip().split("\n"):
            match = re.search(r"^\d+\.\s*(.*)", line)
            if match:
                personas.append(match.group(1).strip())

        # Fallback if parsing fails
        if not personas and gen_persona_output:
             personas = [line.strip() for line in gen_persona_output.strip().split('\n') if line.strip()]

        return dspy.Prediction(
            personas=personas,
            raw_personas_output=gen_persona_output, # Keep raw for debugging
            related_sources=related_sources_text
        )

# --- Main Generator Class ---

class RubricPersonaGenerator:
    """
    Generates evaluator personas based on the topic and evaluation context.
    """
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.create_evaluator_persona = CreateRubricEvaluatorPersona(engine=engine)

    def generate_persona(self, topic: str, evaluation_context: str, max_num_persona: int = 3) -> List[str]:
        """
        Generates a list of evaluator personas. Includes a default 'General Evaluator'.
        """
        persona_result = self.create_evaluator_persona(topic=topic, evaluation_context=evaluation_context)
        generated_personas = persona_result.personas

        # Add a default persona for broad coverage
        default_persona = "General Evaluator: Focuses on overall quality, balancing key aspects relevant to the evaluation context."

        # Combine and limit
        considered_personas = [default_persona] + generated_personas
        # Ensure unique personas (basic check)
        considered_personas = list(dict.fromkeys(considered_personas))

        return considered_personas[:max_num_persona + 1] # +1 for the default