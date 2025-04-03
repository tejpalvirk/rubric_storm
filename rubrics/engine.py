import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Union, Literal, Optional, Dict

import dspy

# Import rubric-specific components
from .rubric_dataclass import Rubric
from .persona_generator import RubricPersonaGenerator
from .knowledge_curation import RubricKnowledgeCurationModule
from .criteria_generation import RubricCriteriaGenerationModule
from .condition_generation import RubricConditionGenerationModule
from .rubric_polish import RubricPolishingModule

# Import core components (adjust paths as necessary)
from ..interface import Engine, LMConfigs, Retriever
from ..lm import LitellmModel # Example LM
from ..utils import FileIOHelper, makeStringRed, truncate_filename
# Assuming StormInformationTable is used or adapted
from ..storm_wiki.modules.storm_dataclass import StormInformationTable

logger = logging.getLogger(__name__)

@dataclass
class RubricRunnerArguments:
    """Arguments for controlling the Rubric generation pipeline."""
    output_dir: str = field(
        metadata={"help": "Output directory for the results."}
    )
    evaluation_context: str = field(
        metadata={"help": "Specific context or goal for the evaluation rubric."}
    )
    base_score: int = field(
        default=50,
        metadata={"help": "Starting score before applying condition point deltas."}
    )
    # Inherit relevant args from STORM or redefine
    max_conv_turn: int = field(
        default=3,
        metadata={"help": "Maximum number of dialogue turns per persona."}
    )
    max_perspective: int = field(
        default=3,
        metadata={"help": "Maximum number of evaluator personas to use."}
    )
    max_search_queries_per_turn: int = field(
        default=3,
        metadata={"help": "Maximum search queries per question."}
    )
    disable_perspective: bool = field(
        default=False,
        metadata={"help": "If True, use only the 'General Evaluator' persona."}
    )
    search_top_k: int = field(
        default=5, # May need more snippets for condition generation
        metadata={"help": "Top k search results per query."}
    )
    max_thread_num: int = field(
        default=5, # Adjust based on API limits and system resources
        metadata={"help": "Maximum number of threads for parallel operations."}
    )
    # Add any other relevant arguments from STORMWikiRunnerArguments if needed

    def to_dict(self):
        return asdict(self)


class RubricLMConfigs(LMConfigs):
    """Configurations for LMs used in different parts of Rubric generation."""
    def __init__(self):
        super().__init__() # Initialize parent if it has relevant methods
        self.persona_gen_lm: Optional[Union[dspy.dsp.LM, dspy.dsp.HFModel]] = None
        self.knowledge_curation_lm: Optional[Union[dspy.dsp.LM, dspy.dsp.HFModel]] = None # For expert answers
        self.question_asker_lm: Optional[Union[dspy.dsp.LM, dspy.dsp.HFModel]] = None # For seeker questions
        self.criteria_gen_lm: Optional[Union[dspy.dsp.LM, dspy.dsp.HFModel]] = None
        self.condition_gen_lm: Optional[Union[dspy.dsp.LM, dspy.dsp.HFModel]] = None # Might need multiple for sub-tasks
        self.rubric_polish_lm: Optional[Union[dspy.dsp.LM, dspy.dsp.HFModel]] = None

    # Add setter methods like in STORMWikiLMConfigs
    def set_persona_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]): self.persona_gen_lm = model
    def set_knowledge_curation_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]): self.knowledge_curation_lm = model
    def set_question_asker_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]): self.question_asker_lm = model
    def set_criteria_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]): self.criteria_gen_lm = model
    def set_condition_gen_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]): self.condition_gen_lm = model
    def set_rubric_polish_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]): self.rubric_polish_lm = model

    # init_check, collect_and_reset_lm_history, collect_and_reset_lm_usage, log methods
    # can potentially be inherited or reimplemented if needed.


class RubricRunner(Engine):
    """Rubric generation pipeline runner."""

    def __init__(self, args: RubricRunnerArguments, lm_configs: RubricLMConfigs, rm: dspy.Retrieve):
        super().__init__(lm_configs=lm_configs) # Pass lm_configs to parent
        self.args = args
        # self.lm_configs = lm_configs # Already set in parent
        self.retriever = Retriever(rm=rm, max_thread=self.args.max_thread_num) # Use the passed RM

        # Instantiate Rubric-specific modules
        rubric_persona_generator = RubricPersonaGenerator(
            engine=self.lm_configs.persona_gen_lm
        )
        self.knowledge_curation_module = RubricKnowledgeCurationModule(
            retriever=self.retriever, # Pass the retriever instance
            persona_generator=rubric_persona_generator,
            conv_simulator_lm=self.lm_configs.knowledge_curation_lm,
            question_asker_lm=self.lm_configs.question_asker_lm,
            max_search_queries_per_turn=self.args.max_search_queries_per_turn,
            max_conv_turn=self.args.max_conv_turn,
            max_thread_num=self.args.max_thread_num,
        )
        self.criteria_generation_module = RubricCriteriaGenerationModule(
            criteria_gen_lm=self.lm_configs.criteria_gen_lm
        )
        self.condition_generation_module = RubricConditionGenerationModule(
             condition_gen_lm=self.lm_configs.condition_gen_lm
        )
        self.rubric_polishing_module = RubricPolishingModule(
             rubric_polish_lm=self.lm_configs.rubric_polish_lm
        )

        self.lm_configs.init_check() # Check if all LMs are set
        # self.apply_decorators() # Apply decorators if defined in parent Engine

    # --- Placeholder Methods for Loading Intermediate Results ---
    # These need actual implementation based on how/if information_table and rubric objects are saved.

    def _load_information_table(self, file_path: str) -> Optional[StormInformationTable]:
        logger.info(f"Attempting to load information table from: {file_path}")
        if os.path.exists(file_path):
            try:
                # Assuming StormInformationTable has a suitable loading method
                return StormInformationTable.from_conversation_log_file(file_path)
            except Exception as e:
                logger.error(f"Failed to load information table from {file_path}: {e}")
                return None
        logger.warning(f"Information table file not found: {file_path}")
        return None

    def _load_rubric(self, file_path: str) -> Optional[Rubric]:
         logger.info(f"Attempting to load rubric from: {file_path}")
         if os.path.exists(file_path):
            try:
                return Rubric.load_from_json(file_path)
            except Exception as e:
                logger.error(f"Failed to load rubric from {file_path}: {e}")
                return None
         logger.warning(f"Rubric file not found: {file_path}")
         return None


    # --- Core Pipeline Execution ---

    def run(
        self,
        topic: str,
        evaluation_context: str, # Added context
        # Add do_* flags similar to STORM
        do_research: bool = True,
        do_generate_criteria: bool = True,
        do_generate_conditions: bool = True,
        do_polish_rubric: bool = True,
        # ground_truth_url: Optional[str] = None, # If needed for excluding sources
        # callback_handler: Optional[BaseCallbackHandler] = None, # If callbacks are used
    ):
        """Runs the rubric generation pipeline."""

        self.topic = topic
        self.evaluation_context = evaluation_context

        # Sanitize topic and context for directory name
        topic_sanitized = truncate_filename(topic.replace(" ", "_").replace("/", "_"))
        context_sanitized = truncate_filename(evaluation_context.replace(" ", "_").replace("/", "_"))[:50] # Limit length
        self.output_dir_specific = os.path.join(self.args.output_dir, f"{topic_sanitized}__{context_sanitized}")
        os.makedirs(self.output_dir_specific, exist_ok=True)

        logger.info(f"Starting rubric generation for Topic: '{topic}', Context: '{evaluation_context}'")
        logger.info(f"Output directory: {self.output_dir_specific}")

        # --- Stage 1: Knowledge Curation ---
        information_table: Optional[StormInformationTable] = None
        info_table_path = os.path.join(self.output_dir_specific, "conversation_log.json") # Assuming same format
        if do_research:
            logger.info("Starting Stage 1: Knowledge Curation...")
            # Note: The research method needs evaluation_context
            info_table_result = self.knowledge_curation_module.research(
                topic=self.topic,
                evaluation_context=self.evaluation_context,
                # ground_truth_url=ground_truth_url,
                max_perspective=self.args.max_perspective,
                disable_perspective=self.args.disable_perspective,
                # callback_handler=callback_handler,
                return_conversation_log=True # Get log for saving
            )
            information_table, conv_log = info_table_result
            # Save conversation log
            FileIOHelper.dump_json(conv_log, info_table_path)
            # Save raw search results (if method available)
            try:
                 raw_results_path = os.path.join(self.output_dir_specific, "raw_search_results.json")
                 information_table.dump_url_to_info(raw_results_path)
            except AttributeError:
                 logger.warning("InformationTable does not have dump_url_to_info method.")
            logger.info("Finished Stage 1.")
        else:
            logger.info("Skipping Stage 1: Loading existing information table...")
            information_table = self._load_information_table(info_table_path)
            if information_table is None:
                logger.error(f"Failed to load information table. Cannot proceed without it for later stages.")
                return

        # --- Stage 2: Criteria Generation ---
        rubric_with_criteria: Optional[Rubric] = None
        criteria_rubric_path = os.path.join(self.output_dir_specific, "rubric_criteria_gen.json")
        if do_generate_criteria:
            logger.info("Starting Stage 2: Criteria Generation...")
            if information_table is None: # Should have been loaded if do_research=False
                 logger.error("Information table is missing. Cannot generate criteria.")
                 return
            rubric_with_criteria = self.criteria_generation_module.forward(
                topic=self.topic,
                evaluation_context=self.evaluation_context,
                information_table=information_table
            )
            rubric_with_criteria.save_to_json(criteria_rubric_path)
            logger.info(f"Finished Stage 2. Criteria generated: {[c.name for c in rubric_with_criteria.criteria]}")
        else:
            logger.info("Skipping Stage 2: Loading existing rubric with criteria...")
            rubric_with_criteria = self._load_rubric(criteria_rubric_path)
            if rubric_with_criteria is None:
                 logger.error("Failed to load rubric with criteria. Cannot proceed.")
                 return


        # --- Stage 3: Condition Generation ---
        rubric_with_conditions: Optional[Rubric] = None
        conditions_rubric_path = os.path.join(self.output_dir_specific, "rubric_conditions_gen.json")
        if do_generate_conditions:
             logger.info("Starting Stage 3: Condition Generation...")
             if rubric_with_criteria is None:
                 logger.error("Rubric with criteria is missing. Cannot generate conditions.")
                 return
             if information_table is None: # Need info table for grounding examples
                 logger.warning("Information table missing, condition examples might lack grounding.")
                 # Attempt to load it again if possible
                 information_table = self._load_information_table(info_table_path)
                 if information_table is None:
                     logger.error("Cannot proceed without information table for condition generation.")
                     return

             rubric_with_conditions = self.condition_generation_module.forward(
                 rubric=rubric_with_criteria, # Pass rubric with criteria structure
                 information_table=information_table
             )
             rubric_with_conditions.save_to_json(conditions_rubric_path)
             logger.info("Finished Stage 3.")
        else:
             logger.info("Skipping Stage 3: Loading existing rubric with conditions...")
             rubric_with_conditions = self._load_rubric(conditions_rubric_path)
             if rubric_with_conditions is None:
                 logger.error("Failed to load rubric with conditions. Cannot proceed.")
                 return

        # --- Stage 4: Rubric Polishing ---
        final_rubric: Optional[Rubric] = None
        final_rubric_json_path = os.path.join(self.output_dir_specific, "final_rubric.json")
        final_rubric_md_path = os.path.join(self.output_dir_specific, "final_rubric.md")
        if do_polish_rubric:
            logger.info("Starting Stage 4: Rubric Polishing...")
            if rubric_with_conditions is None:
                logger.error("Rubric with conditions is missing. Cannot polish.")
                return
            final_rubric = self.rubric_polishing_module.forward(
                rubric=rubric_with_conditions
            )
            final_rubric.save_to_json(final_rubric_json_path)
            final_rubric.save_to_markdown(final_rubric_md_path)
            logger.info("Finished Stage 4. Final rubric saved.")
        else:
             logger.info("Skipping Stage 4: Polishing.")
             final_rubric = rubric_with_conditions # Use the unpolished version
             if final_rubric: # Save if loaded/generated
                 final_rubric.save_to_json(final_rubric_json_path)
                 final_rubric.save_to_markdown(final_rubric_md_path)

        logger.info("Rubric generation pipeline finished.")


    def post_run(self):
        """Logs run configuration and potentially LLM history."""
        # Dump run configuration
        config_log = {
             "runner_args": self.args.to_dict(),
             "lm_configs": self.lm_configs.log() # Assuming log method exists
        }
        config_path = os.path.join(self.output_dir_specific, "run_config.json")
        FileIOHelper.dump_json(config_log, config_path)
        logger.info(f"Run configuration saved to {config_path}")

        # Dump LLM history (optional, might be large)
        # llm_history = self.lm_configs.collect_and_reset_lm_history() # Assuming method exists
        # history_path = os.path.join(self.output_dir_specific, "llm_history.jsonl")
        # try:
        #     with open(history_path, "w") as f:
        #         for call in llm_history:
        #             f.write(json.dumps(call) + "\n")
        #     logger.info(f"LLM history saved to {history_path}")
        # except Exception as e:
        #      logger.error(f"Failed to save LLM history: {e}")

    # summary method might need adaptation if cost tracking changes
    # def summary(self):
    #     print("--- Rubric Generation Summary ---")
    #     # Adapt parent's summary or reimplement
    #     super().summary()