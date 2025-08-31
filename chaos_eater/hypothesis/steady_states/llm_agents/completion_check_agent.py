from typing import Dict, Tuple

from ....preprocessing.preprocessor import ProcessedData
from ....utils.wrappers import LLM, BaseModel, Field
from ....utils.llms import build_json_agent, LoggingCallback, LLMLog, safe_get_response_field
from ....utils.functions import StreamDebouncer, MessageLogger


#---------
# prompts
#---------
SYS_CHECK_STEADY_STATE_COMPLETION = """\
You are a helpful AI assistant for Chaos Engineering.
Given K8s manifests for a system, user's instructions, and steady states already defined, you will determine whether an additional steady state needs to be defined.
Always keep the following rules:
- Clearly describe the reason for determining whether an additional steady state is needed.
- You may also cite the user's instructions as the reason.
- {format_instructions}"""

USER_CHECK_STEADY_STATE_COMPLETION = """\
# Here is the overview of my system:
{user_input}

# Please follow the instructions below regarding Chaos Engineering:
{ce_instructions}

# Steady states already defined are as follows:
{predefined_steady_states}

Now, determine whether an additional steady state needs to be defined."""

#--------------------
# json output format
#--------------------
class SteadyStateCompletionCheck(BaseModel):
    thought: str = Field(description="Describe your thought process of determing whether an additional steady states is needed.")
    requires_addition: bool = Field(description="The necessity of an additional steady state. If it is needed, select 'True'; otherwise select 'False'.")

#------------------
# agent definition
#------------------
class SteadyStateCompletionCheckAgent:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.agent = build_json_agent(
            llm=llm,
            chat_messages=[("system", SYS_CHECK_STEADY_STATE_COMPLETION), ("human", USER_CHECK_STEADY_STATE_COMPLETION)],
            pydantic_object=SteadyStateCompletionCheck,
            is_async=False
        )

    def check_steady_state_completion(
        self,
        input_data: ProcessedData,
        predefined_steady_states: list,
        message_logger: MessageLogger
    ) -> Tuple[LLMLog, Dict[str, str]]:
        logger = LoggingCallback(name="steady_state_completion_check", llm=self.llm)
        debouncer = StreamDebouncer()
        container = message_logger.container(border=True)
        container.write("##### Steady state completion check")
        thought_empty = container.placeholder()
        check_empty = container.placeholder()

        def display_responce(responce) -> None:
            # Use safe field extraction to handle different response formats
            thought = safe_get_response_field(responce, "thought")
            check = safe_get_response_field(responce, "requires_addition")
            
            if thought is not None:
                thought_empty.write(thought)
            if check is not None:
                check_empty.write(f"An additional steady state is needed?: ```{check}```")

        for completion_check in self.agent.stream({
            "user_input": input_data.to_k8s_overview_str(), 
            "ce_instructions": input_data.ce_instructions,
            "predefined_steady_states": predefined_steady_states.to_str()},
            {"callbacks": [logger]}
        ):
            if debouncer.should_update():
                display_responce(completion_check)
        display_responce(completion_check)
        return logger.log, completion_check