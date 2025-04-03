import logging
import re
import concurrent.futures
from concurrent.futures import as_completed
from typing import Union, List, Optional, Dict, Tuple
import json # For JSON parsing if needed

import dspy

# Use the adapted data classes
# Assuming Information is imported from interface or dataclass
from ...interface import Information, Retriever, KnowledgeCurationModule
from ...storm_wiki.modules.storm_dataclass import DialogueTurn # Can keep using this if structure is fine
from ...utils import ArticleTextProcessing
from .persona_generator import RubricPersonaGenerator
# Assuming BaseCallbackHandler is defined elsewhere (e.g., in collaborative_storm or a shared module)
from ...collaborative_storm.modules.callback import BaseCallbackHandler

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    streamlit_connection = True
except ImportError:
    streamlit_connection = False

# --- dspy Signatures (Adapted for Rubrics) ---

class AskRubricQuestionWithPersona(dspy.Signature):
    """
    You are an evaluator preparing to create a detailed rubric. Your current persona focuses on a specific aspect of the evaluation.
    Based on the conversation history with an expert about the topic and evaluation context, ask a targeted question to uncover specific details needed for the rubric.
    Focus on:
    - Identifying specific, observable positive/negative attributes.
    - Finding concrete examples of different quality levels or errors.
    - Understanding how to differentiate minor vs. major issues for point assignment.
    - Clarifying criteria relevant to your persona's focus.

    Avoid asking questions already covered. If you have enough information for your persona's focus, say "Thank you, I have enough details on this aspect for now."
    """
    topic = dspy.InputField(prefix="Topic being evaluated:")
    evaluation_context = dspy.InputField(prefix="Specific evaluation context/goal:")
    persona = dspy.InputField(prefix="Your current evaluation persona/focus:")
    conv_history = dspy.InputField(prefix="Conversation history (summary):", format=str)
    question = dspy.OutputField(prefix="Your next question for the expert:", format=str)


class AskRubricQuestion(dspy.Signature):
    """
    You are an evaluator preparing to create a detailed rubric.
    Based on the conversation history with an expert about the topic and evaluation context, ask a targeted question to uncover specific details needed for the rubric.
    Focus on:
    - Identifying specific, observable positive/negative attributes.
    - Finding concrete examples of different quality levels or errors.
    - Understanding how to differentiate minor vs. major issues for point assignment.
    - Clarifying evaluation criteria.

    Avoid asking questions already covered. If you feel the major aspects are covered, say "Thank you, I think I have a good overview now."
    """
    topic = dspy.InputField(prefix="Topic being evaluated:")
    evaluation_context = dspy.InputField(prefix="Specific evaluation context/goal:")
    conv_history = dspy.InputField(prefix="Conversation history (summary):", format=str)
    question = dspy.OutputField(prefix="Your next question for the expert:", format=str)


class RubricQuestionToQuery(dspy.Signature):
    """
    You need to find information online to answer a specific question relevant to creating an evaluation rubric.
    The rubric is for the given topic and evaluation context.
    Generate specific search queries to find: evaluation standards, comparison guides, user reviews, expert analyses, technical specs, case studies, or examples of best/worst practices relevant to the question.
    Output queries, one per line, prefixed with '- '.
    """
    topic = dspy.InputField(prefix="Topic being evaluated:")
    evaluation_context = dspy.InputField(prefix="Specific evaluation context/goal:")
    question = dspy.InputField(prefix="Question to research for the rubric:")
    queries = dspy.OutputField(prefix="Search Queries:\n", format=str)


class AnswerRubricQuestion(dspy.Signature):
    """
    You are an expert answering questions to help build an evaluation rubric for the Topic in the specific Evaluation Context.
    Use the Gathered Information to provide a concise, factual answer to the Question.
    Highlight specific attributes, conditions, examples, or data points relevant to evaluation.
    Cite sources using [1], [2], etc. Do not add a bibliography.
    If the information is insufficient, state that clearly.
    """
    topic = dspy.InputField(prefix="Topic being evaluated:")
    evaluation_context = dspy.InputField(prefix="Specific evaluation context/goal:")
    question = dspy.InputField(prefix="Question asked:")
    gathered_info = dspy.InputField(prefix="Gathered Information:\n", format=str)
    answer = dspy.OutputField(prefix="Expert Answer:\n", format=str)

# --- dspy Modules (Adapted for Rubrics) ---

class RubricCriteriaSeeker(dspy.Module):
    """Asks questions from a specific evaluator persona perspective."""
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        # Using ChainOfThought for potentially more nuanced question asking
        self.ask_question_with_persona = dspy.ChainOfThought(AskRubricQuestionWithPersona)
        self.ask_question = dspy.ChainOfThought(AskRubricQuestion)
        self.engine = engine

    def forward(self, topic: str, evaluation_context: str, persona: str, dialogue_turns: List[DialogueTurn]):
        # Summarize conversation history (similar to STORM)
        conv_summary = []
        limit = 4 # Show more recent turns in detail
        for i, turn in enumerate(reversed(dialogue_turns)):
             # Limit summary length
             if i >= limit * 2 : break # Limit total turns considered
             expert_answer = ArticleTextProcessing.remove_citations(turn.agent_utterance)
             if i < limit : # Show recent turns fully
                 conv_summary.append(f"Expert: {expert_answer}\nSeeker ({persona}): {turn.user_utterance}")
             else: # Summarize older turns
                  conv_summary.append(f"Seeker ({persona}): {turn.user_utterance}\nExpert: [Answer Summary Omitted]")

        conv_summary_str = "\n---\n".join(reversed(conv_summary)) # Chronological order
        conv_summary_str = conv_summary_str.strip() or "N/A"
        # Limit overall length
        conv_summary_str = ArticleTextProcessing.limit_word_count_preserve_newline(conv_summary_str, 2000)


        with dspy.settings.context(lm=self.engine):
            if persona and persona.strip() and "General Evaluator" not in persona:
                question = self.ask_question_with_persona(
                    topic=topic,
                    evaluation_context=evaluation_context,
                    persona=persona,
                    conv_history=conv_summary_str
                ).question
            else:
                # Use general question asking if no specific persona or it's the default
                question = self.ask_question(
                    topic=topic,
                    evaluation_context=evaluation_context,
                    conv_history=conv_summary_str
                ).question

        return dspy.Prediction(question=question.strip())


class RubricTopicExpert(dspy.Module):
    """Answers questions by retrieving information and synthesizing."""
    def __init__(
        self,
        engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retriever: Retriever,
        max_search_queries: int,
    ):
        super().__init__()
        self.generate_queries = dspy.Predict(RubricQuestionToQuery)
        self.retriever = retriever
        self.answer_question = dspy.Predict(AnswerRubricQuestion)
        self.engine = engine
        self.max_search_queries = max_search_queries

    def forward(self, topic: str, evaluation_context: str, question: str, ground_truth_url: Optional[str] = None):
        with dspy.settings.context(lm=self.engine):
            # Generate search queries
            queries_result = self.generate_queries(
                topic=topic, evaluation_context=evaluation_context, question=question
            )
            queries_str = queries_result.queries
            queries = [q.strip("- ").strip() for q in queries_str.strip().split("\n") if q.strip()]
            queries = queries[: self.max_search_queries]

            # Retrieve search results
            exclude_urls = [ground_truth_url] if ground_truth_url else []
            searched_results: List[Information] = self.retriever.forward(queries, exclude_urls=exclude_urls)

            # Prepare context for answering
            if searched_results:
                # Format snippets for the LLM - similar to STORM
                info_context = ""
                idx_to_info_map = {}
                current_word_count = 0
                max_words = 2500 # Limit context size
                for idx, info in enumerate(searched_results):
                    if not info.snippets: continue
                    snippet = info.snippets[0] # Use first snippet for brevity
                    snippet_len = len(snippet.split())
                    if current_word_count + snippet_len > max_words:
                        break
                    info_line = f"[{idx+1}] Source: {info.title} ({info.url})\nSnippet: {snippet}\n\n"
                    info_context += info_line
                    idx_to_info_map[idx + 1] = info # Map original index to info object
                    current_word_count += snippet_len + len(info.title.split()) # Approx count

                if not info_context:
                     answer = "I found some sources, but could not extract relevant snippets to answer the question."
                     cited_results = {}
                else:
                    # Generate answer
                    answer_result = self.answer_question(
                        topic=topic,
                        evaluation_context=evaluation_context,
                        question=question,
                        gathered_info=info_context.strip()
                    )
                    answer = answer_result.answer

                    # Process citations (similar to STORM)
                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(answer)
                    # Extract cited indices from the generated answer
                    cited_indices = set(map(int, re.findall(r"\[(\d+)\]", answer)))
                    # Map answer indices back to the original searched_results list
                    cited_results = {
                         answer_idx: idx_to_info_map[original_idx]
                         for answer_idx, original_idx in enumerate(cited_indices, 1) # Assuming answer uses 1-based indexing matching the prompt
                         if original_idx in idx_to_info_map
                     }
                     # Optional: Re-index citations in the answer text if needed, similar to StormArticle update_citation_index

            else:
                answer = "I could not find relevant information online to answer this question."
                cited_results = {} # No cited results

        # Keep track of raw results for potential later use/analysis
        raw_search_results_for_turn = [info.to_dict() for info in searched_results]

        return dspy.Prediction(
            answer=answer.strip(),
            queries=queries,
            # Store raw results for logging/traceability
            searched_results_raw=raw_search_results_for_turn,
             # Store only the *cited* Information objects mapped to their citation index in the answer
            cited_results_map=cited_results
        )


class RubricConvSimulator(dspy.Module):
    """Simulates an evaluation rubric building conversation."""
    def __init__(
        self,
        topic_expert_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        question_asker_engine: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        retriever: Retriever,
        max_search_queries_per_turn: int,
        max_conv_turn: int,
    ):
        super().__init__()
        self.criteria_seeker = RubricCriteriaSeeker(engine=question_asker_engine)
        self.topic_expert = RubricTopicExpert(
            engine=topic_expert_engine,
            retriever=retriever,
            max_search_queries=max_search_queries_per_turn,
        )
        self.max_turn = max_conv_turn

    def forward(
        self,
        topic: str,
        evaluation_context: str,
        persona: str,
        ground_truth_url: Optional[str] = None,
        callback_handler: Optional[BaseCallbackHandler] = None, # Adjusted for optional handler
    ) -> dspy.Prediction:
        """
        Simulates conversation.
        Returns:
            dspy.Prediction: Contains 'dlg_history' (List[DialogueTurn]).
        """
        dlg_history: List[DialogueTurn] = []
        logging.info(f"Starting conversation for persona: {persona}")

        for turn_num in range(self.max_turn):
            logging.debug(f"Turn {turn_num + 1}, Persona: {persona}")
            # Seeker asks a question
            seeker_question = self.criteria_seeker(
                topic=topic,
                evaluation_context=evaluation_context,
                persona=persona,
                dialogue_turns=dlg_history
            ).question

            logging.debug(f"Seeker question: {seeker_question}")
            if not seeker_question or seeker_question.startswith("Thank you"):
                logging.info(f"Ending conversation for persona {persona} due to seeker concluding.")
                break

            # Expert answers
            expert_output = self.topic_expert(
                topic=topic,
                evaluation_context=evaluation_context,
                question=seeker_question,
                ground_truth_url=ground_truth_url
            )
            logging.debug(f"Expert answer: {expert_output.answer[:100]}...") # Log snippet

            # Create DialogueTurn
            # Note: searched_results now stores the *cited* results map
            # We might need raw results in the log for full traceability
            dlg_turn = DialogueTurn(
                agent_utterance=expert_output.answer,
                user_utterance=seeker_question,
                search_queries=expert_output.queries,
                 # Storing raw results for logging, cited results for potential use
                search_results=expert_output.searched_results_raw
            )
            # Add cited map info separately if needed for analysis later
            # dlg_turn.cited_results_map = expert_output.cited_results_map

            dlg_history.append(dlg_turn)
            if callback_handler:
                try:
                    callback_handler.on_dialogue_turn_end(dlg_turn=dlg_turn)
                except Exception as e:
                    logging.error(f"Error in callback handler: {e}")

        logging.info(f"Finished conversation for persona: {persona} after {len(dlg_history)} turns.")
        return dspy.Prediction(dlg_history=dlg_history)


class RubricKnowledgeCurationModule(KnowledgeCurationModule):
    """Knowledge Curation module adapted for Rubric generation."""
    def __init__(
        self,
        retriever: Retriever,
        persona_generator: RubricPersonaGenerator,
        conv_simulator_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        question_asker_lm: Union[dspy.dsp.LM, dspy.dsp.HFModel],
        max_search_queries_per_turn: int,
        max_conv_turn: int,
        max_thread_num: int,
    ):
        super().__init__(retriever=retriever) # Pass retriever to parent
        self.persona_generator = persona_generator
        self.max_thread_num = max_thread_num
        self.conv_simulator = RubricConvSimulator(
            topic_expert_engine=conv_simulator_lm,
            question_asker_engine=question_asker_lm,
            retriever=retriever,
            max_search_queries_per_turn=max_search_queries_per_turn,
            max_conv_turn=max_conv_turn,
        )

    def _get_considered_personas(self, topic: str, evaluation_context: str, max_num_persona: int) -> List[str]:
        # Generate personas based on topic and context
        return self.persona_generator.generate_persona(
            topic=topic, evaluation_context=evaluation_context, max_num_persona=max_num_persona
        )

    def _run_conversations(
        self,
        topic: str,
        evaluation_context: str,
        ground_truth_url: Optional[str],
        considered_personas: List[str],
        callback_handler: Optional[BaseCallbackHandler],
    ) -> List[Tuple[str, List[DialogueTurn]]]:
        """Runs conversations concurrently for different personas."""
        conversations = []

        def run_conv(persona):
             # Simulate one conversation for a given persona
             return self.conv_simulator(
                 topic=topic,
                 evaluation_context=evaluation_context,
                 persona=persona,
                 ground_truth_url=ground_truth_url,
                 callback_handler=callback_handler,
             )

        max_workers = min(self.max_thread_num, len(considered_personas))
        logging.info(f"Running {len(considered_personas)} conversations with max {max_workers} workers.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_persona = {
                executor.submit(run_conv, persona): persona
                for persona in considered_personas
            }

            if streamlit_connection:
                for t in executor._threads:
                     add_script_run_ctx(t) # Propagate context if using Streamlit

            for future in as_completed(future_to_persona):
                persona = future_to_persona[future]
                try:
                    conv_result = future.result()
                    # Assuming clean_up_citation is still relevant or needs adaptation
                    # Clean up citations if needed (may need adjustment for rubrics)
                    # cleaned_history = ArticleTextProcessing.clean_up_citation(conv_result).dlg_history
                    cleaned_history = conv_result.dlg_history # Skipping cleanup for now
                    conversations.append((persona, cleaned_history))
                    logging.info(f"Completed conversation for persona: {persona}")
                except Exception as e:
                    logging.error(f"Error running conversation for persona {persona}: {e}", exc_info=True)

        return conversations

    def research(
        self,
        topic: str,
        evaluation_context: str, # Added context
        ground_truth_url: Optional[str] = None,
        max_perspective: int = 3,
        disable_perspective: bool = False,
        callback_handler: Optional[BaseCallbackHandler] = None,
        return_conversation_log: bool = False,
    ) -> Union[StormInformationTable, Tuple[StormInformationTable, Dict]]: # TODO: Adapt return type if StormInformationTable changes significantly
        """
        Curate information for rubric generation.
        """
        # 1. Identify Personas based on topic and context
        if callback_handler: callback_handler.on_identify_perspective_start()
        considered_personas = ["General Evaluator"] # Start with default
        if not disable_perspective and self.persona_generator:
            generated_personas = self._get_considered_personas(
                topic=topic, evaluation_context=evaluation_context, max_num_persona=max_perspective
            )
            # Ensure default is not duplicated if generated
            considered_personas = list(dict.fromkeys(generated_personas)) # Use generated ones primarily
        if callback_handler: callback_handler.on_identify_perspective_end(perspectives=considered_personas)

        # 2. Run Conversations
        if callback_handler: callback_handler.on_information_gathering_start()
        conversations = self._run_conversations(
            topic=topic,
            evaluation_context=evaluation_context,
            ground_truth_url=ground_truth_url,
            considered_personas=considered_personas,
            callback_handler=callback_handler,
        )
        if callback_handler: callback_handler.on_information_gathering_end()

        # 3. Aggregate Information (using existing StormInformationTable for now)
        # TODO: Consider adapting StormInformationTable -> RubricInformationTable if needed
        information_table = StormInformationTable(conversations)

        if return_conversation_log:
            conv_log = StormInformationTable.construct_log_dict(conversations)
            return information_table, conv_log
        else:
            return information_table