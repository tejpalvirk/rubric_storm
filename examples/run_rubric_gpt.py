"""
Rubric Generation pipeline powered by GPT-3.5/4 and a search engine (e.g., Bing).
Generates an evaluation rubric for a given topic and context.

Setup:
- Set OPENAI_API_KEY, OPENAI_API_TYPE, etc. in secrets.toml (see knowledge_storm/utils.py).
- Set BING_SEARCH_API_KEY or other retriever keys in secrets.toml.

Output structure:
args.output_dir/
    topic_sanitized__context_sanitized/
        conversation_log.json           # Log of information-seeking conversation
        raw_search_results.json         # Search results used
        rubric_criteria_gen.json        # Rubric after criteria generation
        rubric_conditions_gen.json      # Rubric after condition generation
        final_rubric.json               # Final polished rubric (JSON)
        final_rubric.md                 # Final polished rubric (Markdown)
        run_config.json                 # Configuration used for the run
        # llm_history.jsonl             # Optional: Log of LLM calls
"""

import os
import sys
import logging
from argparse import ArgumentParser

# Ensure the knowledge_storm package is in the Python path
# If running from root directory, this might not be needed depending on setup
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(parent_dir)

from knowledge_storm.rubrics.engine import (
    RubricRunnerArguments,
    RubricLMConfigs,
    RubricRunner,
)
# Assuming LM classes are accessible from knowledge_storm.lm
from knowledge_storm.lm import OpenAIModel, AzureOpenAIModel, LitellmModel
# Assuming RM classes are accessible
from knowledge_storm.rm import (
    BingSearch, YouRM, BraveRM, SerperRM, DuckDuckGoSearchRM, TavilySearchRM, SearXNG
)
from knowledge_storm.utils import load_api_key

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main(args):
    load_api_key(toml_file_path="secrets.toml")
    lm_configs = RubricLMConfigs()

    # --- LM Configuration (Example using OpenAI/Azure GPT) ---
    # Select Model Class based on API type
    ModelProvider = os.getenv("OPENAI_API_TYPE", "openai")
    ModelClass = LitellmModel # Defaulting to Litellm for flexibility

    # Define model names
    # Adjust based on availability and preference (e.g., use stronger models for generation)
    model_map = {
        "openai": {"fast": "gpt-4o-mini", "strong": "gpt-4o"},
        "azure": {"fast": "azure/gpt-4o-mini", "strong": "azure/gpt-4o"} # Example names, adjust to deployment
    }
    fast_model_name = model_map.get(ModelProvider, {}).get("fast", "gpt-4o-mini")
    strong_model_name = model_map.get(ModelProvider, {}).get("strong", "gpt-4o")

    # Configure kwargs for the selected provider
    lm_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY") if ModelProvider == "openai" else os.getenv("AZURE_API_KEY"),
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if ModelProvider == "azure":
        lm_kwargs["api_base"] = os.getenv("AZURE_API_BASE")
        lm_kwargs["api_version"] = os.getenv("AZURE_API_VERSION")
        # Litellm needs provider specified for Azure
        lm_kwargs["model"] = lm_kwargs["model"].replace("azure/", "") # Remove prefix for litellm model string if using azure
        lm_kwargs["litellm_params"] = {"api_base": lm_kwargs.pop("api_base"), "api_version": lm_kwargs.pop("api_version")}


    # Assign LMs to roles (Example: use faster model for simpler tasks)
    lm_configs.set_persona_gen_lm(ModelClass(model=fast_model_name, max_tokens=400, **lm_kwargs))
    lm_configs.set_question_asker_lm(ModelClass(model=fast_model_name, max_tokens=300, **lm_kwargs))
    lm_configs.set_knowledge_curation_lm(ModelClass(model=fast_model_name, max_tokens=1000, **lm_kwargs)) # Expert answers
    lm_configs.set_criteria_gen_lm(ModelClass(model=strong_model_name, max_tokens=500, **lm_kwargs))
    lm_configs.set_condition_gen_lm(ModelClass(model=strong_model_name, max_tokens=1500, **lm_kwargs)) # Complex task
    lm_configs.set_rubric_polish_lm(ModelClass(model=strong_model_name, max_tokens=2000, **lm_kwargs))
    # --- End LM Configuration ---


    # --- Runner Arguments ---
    runner_args = RubricRunnerArguments(
        output_dir=args.output_dir,
        evaluation_context=args.evaluation_context,
        base_score=args.base_score,
        max_conv_turn=args.max_conv_turn,
        max_perspective=args.max_perspective,
        search_top_k=args.search_top_k,
        max_thread_num=args.max_thread_num,
        max_search_queries_per_turn=args.max_search_queries_per_turn,
        disable_perspective=args.disable_perspective
    )
    # --- End Runner Arguments ---


    # --- Retrieval Module ---
    retriever_choice = args.retriever.lower()
    api_keys = {
         "BING_SEARCH_API_KEY": os.getenv("BING_SEARCH_API_KEY"),
         "YDC_API_KEY": os.getenv("YDC_API_KEY"),
         "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY"),
         "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
         "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
         "SEARXNG_API_KEY": os.getenv("SEARXNG_API_KEY"),
    }

    rm = None
    if retriever_choice == "bing" and api_keys["BING_SEARCH_API_KEY"]:
         rm = BingSearch(bing_search_api_key=api_keys["BING_SEARCH_API_KEY"], k=runner_args.search_top_k)
    elif retriever_choice == "you" and api_keys["YDC_API_KEY"]:
         rm = YouRM(ydc_api_key=api_keys["YDC_API_KEY"], k=runner_args.search_top_k)
    elif retriever_choice == "brave" and api_keys["BRAVE_API_KEY"]:
         rm = BraveRM(brave_search_api_key=api_keys["BRAVE_API_KEY"], k=runner_args.search_top_k)
    elif retriever_choice == "duckduckgo": # No API key needed
         rm = DuckDuckGoSearchRM(k=runner_args.search_top_k)
    elif retriever_choice == "serper" and api_keys["SERPER_API_KEY"]:
         rm = SerperRM(serper_search_api_key=api_keys["SERPER_API_KEY"], k=runner_args.search_top_k)
    elif retriever_choice == "tavily" and api_keys["TAVILY_API_KEY"]:
         rm = TavilySearchRM(tavily_search_api_key=api_keys["TAVILY_API_KEY"], k=runner_args.search_top_k)
    elif retriever_choice == "searxng" and args.searxng_api_url:
         rm = SearXNG(searxng_api_url=args.searxng_api_url, searxng_api_key=api_keys["SEARXNG_API_KEY"], k=runner_args.search_top_k)
    # Add other retrievers similarly...

    if rm is None:
         raise ValueError(f"Could not initialize retriever '{args.retriever}'. Check API keys and arguments.")
    # --- End Retrieval Module ---


    # --- Initialize and Run ---
    runner = RubricRunner(runner_args, lm_configs, rm)

    topic = input("Enter the Topic for the rubric: ")
    # If evaluation context is not passed via args, prompt for it.
    evaluation_context = args.evaluation_context if args.evaluation_context else input("Enter the Evaluation Context: ")
    runner_args.evaluation_context = evaluation_context # Ensure it's set in args

    logger.info(f"Using Topic: {topic}")
    logger.info(f"Using Evaluation Context: {evaluation_context}")

    try:
        runner.run(
            topic=topic,
            evaluation_context=evaluation_context,
            do_research=args.do_research,
            do_generate_criteria=args.do_generate_criteria,
            do_generate_conditions=args.do_generate_conditions,
            do_polish_rubric=args.do_polish_rubric
        )
        runner.post_run()
        # runner.summary() # If summary method is implemented
        logger.info("Rubric generation completed successfully.")
    except Exception as e:
        logger.exception(f"An error occurred during the rubric generation pipeline: {e}")
    # --- End Initialize and Run ---


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate Evaluation Rubrics using STORM adapted pipeline.")

    # Core arguments
    parser.add_argument("--topic", type=str, help="Topic for the rubric (alternative to interactive input).")
    parser.add_argument("--evaluation-context", type=str, required=True, help="Specific evaluation context or goal (e.g., 'Choosing a budget laptop for students').")
    parser.add_argument("--output-dir", type=str, default="./results/rubrics_gpt", help="Directory to store the outputs.")
    parser.add_argument("--retriever", type=str, required=True, choices=["bing", "you", "brave", "serper", "duckduckgo", "tavily", "searxng"], help="Search engine API for information retrieval.")
    parser.add_argument("--searxng_api_url", type=str, help="URL for SearXNG API (required if retriever is searxng).")


    # LM arguments
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for LMs.") # Default lower for rubric gen
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter.")

    # Pipeline stage control
    parser.add_argument("--do-research", action="store_true", default=True, help="Simulate conversation to gather information.")
    parser.add_argument("--skip-research", action="store_false", dest="do_research", help="Skip research phase (load existing data).")
    parser.add_argument("--do-generate-criteria", action="store_true", default=True, help="Generate rubric criteria.")
    parser.add_argument("--skip-criteria", action="store_false", dest="do_generate_criteria", help="Skip criteria generation (load existing).")
    parser.add_argument("--do-generate-conditions", action="store_true", default=True, help="Generate rubric conditions, points, and examples.")
    parser.add_argument("--skip-conditions", action="store_false", dest="do_generate_conditions", help="Skip condition generation (load existing).")
    parser.add_argument("--do-polish-rubric", action="store_true", default=True, help="Polish the final rubric.")
    parser.add_argument("--skip-polish", action="store_false", dest="do_polish_rubric", help="Skip polishing step.")

    # Rubric/STORM hyperparameters
    parser.add_argument("--base-score", type=int, default=50, help="Base score for the rubric.")
    parser.add_argument("--max-conv-turn", type=int, default=3, help="Max dialogue turns per persona.")
    parser.add_argument("--max-perspective", type=int, default=3, help="Max evaluator personas.")
    parser.add_argument("--max-search-queries-per-turn", type=int, default=3, help="Max search queries per question.")
    parser.add_argument("--disable-perspective", action="store_true", help="Use only the general evaluator persona.")
    parser.add_argument("--search-top-k", type=int, default=5, help="Top k search results per query.")
    parser.add_argument("--max-thread-num", type=int, default=5, help="Max threads for parallel tasks.")

    args = parser.parse_args()

    # If topic is provided via argument, use it, otherwise prompt later
    if args.topic:
         # This part needs adjustment if you want non-interactive mode fully
         print(f"Running with Topic: {args.topic}")
         # The input() call in main() needs to be conditional or removed for non-interactive use
         # For now, assumes interactive input if args.topic is not set.
    main(args)
