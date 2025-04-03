# Rubric-STORM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)


**Rubric-STORM** is an LLM system designed to automatically generate detailed evaluation rubrics for a given topic and evaluation context. It adapts the core principles of the original [STORM (Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking)](https://github.com/stanford-oval/storm) system from Stanford, shifting the focus from generating Wikipedia-like articles to creating structured, actionable rubrics suitable for guiding LLM-as-a-judge systems or human evaluators.

The generated rubrics are grounded in information retrieved from external sources (like web search) and consist of specific criteria, conditions with point adjustments (+/- points), and concrete examples, facilitating consistent and detailed evaluations across various domains like products, services, and more.

**Based on:** [STORM Paper (NAACL 2024)](https://arxiv.org/abs/2402.14207) - Please cite the original STORM paper if you use or adapt this work.

## Key Features

*   **Context-Aware Rubric Generation:** Creates rubrics tailored to a specific `topic` and `evaluation_context`.
*   **Condition-Based Structure:** Generates rubrics with specific, observable conditions, each associated with a point delta (+/- points) rather than broad score levels.
*   **Grounded Examples:** Aims to provide concrete examples for conditions, grounded in retrieved information where possible.
*   **Multi-Perspective Information Gathering:** Leverages simulated dialogues with different "evaluator" personas to gather diverse information relevant to criteria and conditions.
*   **Modular Architecture:** Built using `dspy`, allowing for customization of prompts, modules, LMs, and Retrievers.
*   **Extensible:** Supports various Language Models (via `LitellmModel`) and Retrieval Modules (Search Engines, Vector Stores).

## How Rubric-STORM Works

Rubric-STORM adapts the STORM pipeline for rubric generation:

1.  **Information Gathering (Knowledge Curation):**
    *   Given a `topic` and `evaluation_context` (e.g., "Compare budget smartphones for battery life"), the system identifies relevant evaluator personas (e.g., "Tech Reviewer," "Heavy User," "Budget-Conscious Buyer").
    *   It simulates conversations between these personas and a topic expert grounded in external sources (web search, documents).
    *   Questions focus on identifying *evaluation criteria*, *specific positive/negative attributes*, *common pitfalls*, *indicators of quality levels*, and *concrete examples*.
    *   The expert retrieves information and synthesizes answers.

2.  **Rubric Structure Generation (Criteria Generation):**
    *   The system analyzes the gathered information to propose key evaluation criteria (dimensions) relevant to the topic and context (e.g., "Battery Performance," "Camera Quality," "Build Quality").

3.  **Rubric Detail Generation (Condition Generation):**
    *   For each criterion, the system generates specific, observable conditions.
    *   It assigns point deltas (+/- points, defaulting to +/- 1 unless justified) to each condition based on its perceived impact within the evaluation context.
    *   It generates concrete examples illustrating each condition and its associated points, grounding them in the gathered information where feasible.

4.  **Rubric Polishing:**
    *   The system refines the generated rubric for clarity, consistency, and conciseness.
    *   It checks for overlapping or contradictory conditions and verifies example relevance.
    *   It adds overall instructions for the evaluator (LLM or human).

The final output is a structured `Rubric` object containing the criteria, conditions, point deltas, examples, and instructions.

## Installation

### Clone the repository
git clone https://github.com/your-username/rubric-storm.git
cd rubric-storm

### Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### Install dependencies
pip install -r requirements.txt

### Configuration 
Rubric-STORM requires API keys for Language Models and Retrieval Modules. It's recommended to store these in a secrets.toml file in the root directory of your project.

Create secrets.toml with the following format:
```toml
# ============ Language Model Configurations ============
# Example for OpenAI / Azure (using LitellmModel)
OPENAI_API_KEY="your_openai_or_azure_key"
# If using Azure, specify these:
# AZURE_API_BASE="your_azure_endpoint"
# AZURE_API_VERSION="your_api_version"
# OPENAI_API_TYPE="azure" # Set to "openai" or "azure"

# Example for Anthropic
# ANTHROPIC_API_KEY="your_anthropic_key"

# Example for Groq
# GROQ_API_KEY="your_groq_key"

# ... add other LM provider keys as needed

# ============ Retriever Configurations ============
# Provide keys for the retriever you intend to use
BING_SEARCH_API_KEY="your_bing_key"
YDC_API_KEY="your_you_com_key"
SERPER_API_KEY="your_serper_dev_key"
TAVILY_API_KEY="your_tavily_key"
BRAVE_API_KEY="your_brave_search_key"
# SEARXNG_API_KEY="your_searxng_key" # Optional for SearXNG

# Example for Qdrant (if using VectorRM online mode)
# QDRANT_API_KEY="your_qdrant_cloud_key"

# ============ Encoder Configurations (Optional but recommended) ============
# Needed if using VectorRM or future embedding features
ENCODER_API_TYPE="openai" # or "azure", etc. Corresponds to LM keys above.
# AZURE_API_KEY="..." # Encoder might use same or different keys
# AZURE_API_BASE="..."
# AZURE_API_VERSION="..."
```
The load_api_key utility function in the examples loads these into environment variables.

## Basic Usage Example
```python
import os
import logging
from rubric_storm.rubrics.engine import RubricRunner, RubricRunnerArguments, RubricLMConfigs
from rubric_storm.lm import LitellmModel # Example using Litellm
from rubric_storm.rm import BingSearch   # Example using Bing
from rubric_storm.utils import load_api_key

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API keys from secrets.toml
load_api_key(toml_file_path="secrets.toml")

# 1. Configure Language Models
lm_configs = RubricLMConfigs()
# Example: Use GPT-4o-mini for simpler tasks, GPT-4o for complex ones via Litellm
fast_model_name = "gpt-4o-mini"
strong_model_name = "gpt-4o"
common_kwargs = {
    "api_key": os.getenv("OPENAI_API_KEY"), # Assumes OPENAI_API_TYPE is 'openai' or handled by Litellm
    "temperature": 0.7,
    "top_p": 1.0,
}
# Add provider specific details if needed via litellm_params for LitellmModel

lm_configs.set_persona_gen_lm(LitellmModel(model=fast_model_name, max_tokens=400, **common_kwargs))
lm_configs.set_question_asker_lm(LitellmModel(model=fast_model_name, max_tokens=300, **common_kwargs))
lm_configs.set_knowledge_curation_lm(LitellmModel(model=fast_model_name, max_tokens=1000, **common_kwargs)) # Expert answers
lm_configs.set_criteria_gen_lm(LitellmModel(model=strong_model_name, max_tokens=500, **common_kwargs))
lm_configs.set_condition_gen_lm(LitellmModel(model=strong_model_name, max_tokens=1500, **common_kwargs))
lm_configs.set_rubric_polish_lm(LitellmModel(model=strong_model_name, max_tokens=2000, **common_kwargs))

# 2. Configure Retriever Module
bing_key = os.getenv("BING_SEARCH_API_KEY")
if not bing_key:
    raise ValueError("BING_SEARCH_API_KEY not found in environment variables or secrets.toml")
rm = BingSearch(bing_search_api_key=bing_key, k=5) # k=5 for potentially more diverse info

# 3. Set Runner Arguments
topic = "Electric Vehicles (EVs)"
evaluation_context = "Choosing the best EV for city commuting under $40,000, prioritizing range and charging speed."
output_directory = "./my_rubric_output"

runner_args = RubricRunnerArguments(
    output_dir=output_directory,
    evaluation_context=evaluation_context,
    base_score=50,
    max_conv_turn=3,
    max_perspective=3,
    search_top_k=5, # Matches RM 'k'
    max_thread_num=5,
    max_search_queries_per_turn=3,
    disable_perspective=False,
)

# 4. Initialize RubricRunner
runner = RubricRunner(args=runner_args, lm_configs=lm_configs, rm=rm)

# 5. Run the pipeline
try:
    runner.run(
        topic=topic,
        evaluation_context=evaluation_context, # Passed again for clarity, used internally via runner_args
        do_research=True,
        do_generate_criteria=True,
        do_generate_conditions=True,
        do_polish_rubric=True
    )

    # 6. Save logs (optional)
    runner.post_run()
    print(f"Rubric generation complete. Output saved in: {runner.output_dir_specific}")

    # 7. Access the final rubric (optional - runner saves it automatically)
    # final_rubric_path = os.path.join(runner.output_dir_specific, "final_rubric.json")
    # if os.path.exists(final_rubric_path):
    #     from rubric_storm.rubrics.rubric_dataclass import Rubric
    #     final_rubric = Rubric.load_from_json(final_rubric_path)
    #     print("\n--- Generated Rubric (Markdown Preview) ---")
    #     print(final_rubric.to_markdown()[:1000] + "...") # Print first 1000 chars

except Exception as e:
    logging.exception(f"An error occurred during rubric generation: {e}")
```

### Rubric-STORM's modular design allows for customization:

- Language Models: Use any LM supported by dspy or litellm by configuring RubricLMConfigs with the appropriate dspy.dsp.LM instance (e.g., OllamaClient, ClaudeModel, GroqModel). See rubric_storm/lm.py.
- Retrieval Modules: Use different search engines or vector databases by providing an instance of a dspy.Retrieve module (implementations in rubric_storm/rm.py, e.g., YouRM, SerperRM, VectorRM).
- Prompts & Logic: Modify the behavior by editing the dspy.Signature prompts within the module files located in rubric_storm/rubrics/. For example, adjust GenerateConditionExamples in condition_generation.py to change how examples are generated.
 - Pipeline Stages: Modify the RubricRunner.run method in rubric_storm/rubrics/engine.py to change the sequence or logic of pipeline stages.

### Examples
See the examples/rubric_examples/ directory for a runnable script:
run_rubric_gpt.py: Example using GPT models via OpenAI or Azure.

### Contribution
Contributions are welcome! Please feel free to open an issue or submit a pull request for bug fixes, new features, or improved documentation.

## Acknowledgement
This project adapts the core methodology of the STORM system developed by researchers at Stanford University. We are grateful for their foundational work.

If you use Rubric-STORM in your research or work, please cite the original STORM paper:
```bibtex
@inproceedings{shao-etal-2024-assisting,
    title = "Assisting in Writing {W}ikipedia-like Articles From Scratch with Large Language Models",
    author = "Shao, Yijia  and
      Jiang, Yucheng  and
      Kanell, Theodore  and
      Xu, Peter  and
      Khattab, Omar  and
      Lam, Monica",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    publisher = "Association for Computational Linguistics",
}

@inproceedings{jiang-etal-2024-unknown,
    title = "Into the Unknown Unknowns: Engaged Human Learning through Participation in Language Model Agent Conversations",
    author = "Jiang, Yucheng  and
      Shao, Yijia  and
      Ma, Dekun  and
      Semnani, Sina  and
      Lam, Monica",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    year = "2024",
    publisher = "Association for Computational Linguistics",
}
```
(Note: Co-STORM citation included as it builds upon STORM)

License
This project is licensed under the MIT License. 
