# STORM for Rubric Generation

**Note:** This is an adaptation of the STORM system for generating evaluation rubrics, rather than Wikipedia-like articles.

## Overview

This module leverages the core principles and architecture of STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking) to automatically generate detailed evaluation rubrics. Instead of producing a factual narrative about a topic, this system generates structured criteria and conditions suitable for guiding downstream LLM-as-a-judge systems in assigning scores to a given set of options.

**Key Differences from STORM-Wiki:**

*   **Input:** Requires both a `topic` and a specific `evaluation_context` (e.g., "Comparing budget smartphones for battery life", "Evaluating customer service chatbot helpfulness").
*   **Goal:** To produce actionable evaluation criteria, not a comprehensive factual summary.
*   **Output:** A structured `Rubric` object containing criteria, specific conditions, point adjustments (+/- points), and illustrative examples, rather than a Markdown article.
*   **Information Focus:** Knowledge curation targets information relevant to evaluation – quality standards, common errors, distinguishing features, benchmarks, and concrete examples of performance levels.

The generated rubrics are designed to be:

*   **Context-Aware:** Tailored to the specific topic and evaluation goal.
*   **Condition-Based:** Focused on specific, observable attributes or outcomes.
*   **Actionable:** Provide clear point adjustments (+/-) for each condition.
*   **Example-Driven:** Include concrete examples to clarify conditions and point assignments.
*   **Grounded:** Leverage retrieved information (from web search or custom corpora) during generation.

## How it Works

This system adapts the modular STORM pipeline for rubric generation:

1.  **Evaluative Research (Knowledge Curation):**
    *   Given the `topic` and `evaluation_context`, the system identifies relevant evaluator personas (e.g., "Technical Spec Analyst", "End-User Experience Tester", "Cost-Benefit Analyst").
    *   It simulates conversations between these personas and a topic expert (grounded in retrieved information).
    *   Questions focus on identifying key evaluation criteria, specific positive/negative attributes, common pitfalls, differentiating factors, and concrete examples relevant to the evaluation context.
    *   Search queries target sources like comparison sites, reviews, technical docs, user forums, etc.

2.  **Criteria Identification (Outline Generation):**
    *   Analyzes the curated information to identify the main dimensions or criteria for evaluation (e.g., "Performance", "Usability", "Cost", "Accuracy").
    *   Generates a list of `RubricCriterion` objects, each with a name and description.

3.  **Condition & Example Generation (Article Generation):**
    *   For each identified criterion, this stage generates specific, observable conditions.
    *   For each condition, it:
        *   Assigns a point delta (+/- points), typically small (+/- 1) unless a larger impact is justified by the curated information.
        *   Generates concrete examples illustrating the condition and its point consequence, grounded in the retrieved information where possible.
    *   Populates the `conditions` list within each `RubricCriterion` with `RubricCondition` objects.

4.  **Rubric Refinement (Article Polishing):**
    *   Reviews the generated criteria and conditions for clarity, specificity, consistency, and potential overlap.
    *   Refines point deltas and examples for better alignment with the evaluation context.
    *   Adds overall instructions for the LLM judge or human evaluator using the rubric (e.g., handling missing information, score interpretation).

## Data Structures

The primary output is a `Rubric` object, defined in `rubrics/rubric_dataclass.py`. Key components include:

*   **`Rubric`**:
    *   `topic: str`: The overall subject.
    *   `evaluation_context: str`: The specific scenario/goal for the evaluation.
    *   `base_score: int`: The starting score (e.g., 50) before adjustments.
    *   `overall_instructions: Optional[str]`: Guidance for the evaluator.
    *   `criteria: List[RubricCriterion]`: The list of evaluation dimensions.
*   **`RubricCriterion`**:
    *   `name: str`: Name of the criterion (e.g., "Accuracy").
    *   `description: str`: What this criterion covers.
    *   `conditions: List[RubricCondition]`: Specific conditions related to this criterion.
*   **`RubricCondition`**:
    *   `description: str`: A specific, observable condition (e.g., "Contains factual inaccuracy verifiable via provided sources").
    *   `points_delta: int`: Points to add/subtract (e.g., -5).
    *   `examples: List[str]`: Concrete illustrations (e.g., "Example: Claimed capital is Berlin, but source [1] states Paris. [-5 pts]").
    *   `explanation: Optional[str]`: Rationale for points or usage notes.

The `Rubric` object includes methods (`save_to_json`, `save_to_markdown`) for serialization.

## Installation and Setup

This Rubric generation functionality is part of the `knowledge-storm` library.

1.  **Installation:** Follow the main installation instructions for `knowledge-storm` (e.g., `pip install knowledge-storm` or install from source).
2.  **API Keys:** Set up necessary API keys in a `secrets.toml` file in your project root, similar to the main STORM setup. This includes keys for your chosen Language Models (e.g., `OPENAI_API_KEY`) and Retrieval Models (e.g., `BING_SEARCH_API_KEY`).

## Usage

The primary way to use this functionality is via the `RubricRunner` class.

1.  **Import necessary classes:**
    ```python
    from rubrics.engine import RubricRunner, RubricRunnerArguments, RubricLMConfigs
    from lm import LitellmModel # Or your preferred LM class
    from rm import BingSearch # Or your preferred RM class
    from utils import load_api_key
    import os
    ```

2.  **Load API Keys:**
    ```python
    load_api_key(toml_file_path="secrets.toml")
    ```

3.  **Configure LMs (`RubricLMConfigs`):**
    ```python
    lm_configs = RubricLMConfigs()
    # Example using LitellmModel (adapt provider/model names as needed)
    # Use stronger models for generation/polishing
    lm_kwargs = {"api_key": os.getenv("OPENAI_API_KEY"), "temperature": 0.7} # Example
    fast_lm = LitellmModel(model="gpt-4o-mini", max_tokens=500, **lm_kwargs)
    strong_lm = LitellmModel(model="gpt-4o", max_tokens=1500, **lm_kwargs)

    lm_configs.set_persona_gen_lm(fast_lm)
    lm_configs.set_question_asker_lm(fast_lm)
    lm_configs.set_knowledge_curation_lm(fast_lm)
    lm_configs.set_criteria_gen_lm(strong_lm)
    lm_configs.set_condition_gen_lm(strong_lm)
    lm_configs.set_rubric_polish_lm(strong_lm)
    ```

4.  **Configure Runner Arguments (`RubricRunnerArguments`):**
    ```python
    # Essential: Define the evaluation context clearly
    eval_context = "Choosing a budget (<$150) coffee maker for ease of use and brew quality."

    runner_args = RubricRunnerArguments(
        output_dir="./results/rubrics_output",
        evaluation_context=eval_context,
        base_score=50, # Set your desired starting score
        # Adjust other parameters as needed (max_perspective, search_top_k, etc.)
        max_conv_turn=3,
        max_perspective=3,
        search_top_k=5,
        max_thread_num=5,
    )
    ```

5.  **Configure Retriever:**
    ```python
    # Example using Bing Search
    rm = BingSearch(bing_search_api_key=os.getenv("BING_SEARCH_API_KEY"), k=runner_args.search_top_k)
    ```

6.  **Initialize and Run `RubricRunner`:**
    ```python
    runner = RubricRunner(runner_args, lm_configs, rm)

    topic = input("Enter the Topic for the rubric (e.g., 'Budget Coffee Makers'): ")

    runner.run(
        topic=topic,
        evaluation_context=runner_args.evaluation_context, # Use context from args
        # Control pipeline stages (defaults are True)
        # do_research=True,
        # do_generate_criteria=True,
        # do_generate_conditions=True,
        # do_polish_rubric=True
    )

    # Optional: Call post-run hooks if implemented
    runner.post_run()
    ```

7.  **Example Script:** Refer to `examples/rubric_examples/run_rubric_gpt.py` for a complete runnable example with argument parsing.

**Expected Output:**

The runner will create a directory structure like:
`./results/rubrics_output/topic_sanitized__context_sanitized/`
containing:
*   `conversation_log.json`: Logs from the evaluative research phase.
*   `raw_search_results.json`: Aggregated search results.
*   `rubric_criteria_gen.json`: Rubric state after criteria identification.
*   `rubric_conditions_gen.json`: Rubric state after condition/example generation.
*   `final_rubric.json`: The final polished rubric in JSON format.
*   `final_rubric.md`: The final polished rubric in Markdown format for readability.
*   `run_config.json`: Configuration used for the run.

## Customization

The rubric generation pipeline is designed to be modular and customizable:

*   **Prompts (`dspy.Signature`)**: The core logic resides in the prompts defined within the `dspy.Signature` classes in each module (`persona_generator.py`, `knowledge_curation.py`, `criteria_generation.py`, `condition_generation.py`, `rubric_polish.py`). Modifying these prompts is the primary way to change the system's behavior, criteria focus, condition style, example format, etc.
*   **Modules (`dspy.Module`)**: You can replace entire modules (e.g., implement a different condition generation strategy) by creating a new class that adheres to the expected input/output and modifying the `RubricRunner` to use it.
*   **Language Models (`RubricLMConfigs`)**: Easily swap different LMs for different tasks via the `RubricLMConfigs` setters.
*   **Retrieval Models (`rm`)**: Use any `dspy.Retrieve` compatible module (e.g., `YouRM`, `BingSearch`, `VectorRM` for custom corpora) by passing it to the `RubricRunner`.
*   **Data Structures (`rubric_dataclass.py`)**: Modify the `Rubric`, `RubricCriterion`, or `RubricCondition` classes if different output structures are needed.

## Citation

If you use this Rubric generation adaptation, please cite the original STORM papers:

```bibtex
@inproceedings{jiang-etal-2024-unknown,
    title = "Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations",
    # ... (rest of Co-STORM citation) ...
}

@inproceedings{shao-etal-2024-assisting,
    title = "Assisting in Writing {W}ikipedia-like Articles From Scratch with Large Language Models",
    # ... (rest of STORM citation) ...
}
