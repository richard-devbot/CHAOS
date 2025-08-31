from typing import List, Tuple

from ...utils.wrappers import LLM, BaseModel, Field
from ...utils.llms import build_json_agent, LoggingCallback, LLMLog, safe_get_response_field, safe_stream_response_extract
from ...utils.schemas import File
from ...utils.functions import MessageLogger, StreamDebouncer
from typing import Union, Any


#---------
# prompts
#---------
SYS_SUMMARIZE_K8S_WEAKNESSES = """\
You are a professional Kubernetes (K8s) engineer.
Given K8s manifests for a system, you will identify their potential issues related to resiliency and redundancy that may arise during system failures.
Always adhere to the following rules:
- For each issue, provide a name for the issue, the associated K8s manifest(s), description of the potential issues caused by fault injection, and the configuration leading to the issue (no need to suggest improvements).
- If the same issue is present in multiple manifests, merge it into a single issue, specifying all relevant manifest names.
- {format_instructions}"""

USER_SUMMARIZE_K8S_WEAKNESSES = """\
# Here is the K8s manifests for my system.
{k8s_yamls}

Please list issues for each K8s manifest."""


#--------------------
# JSON output format
#--------------------
class K8sIssue(BaseModel):
    issue_name: str = Field(description="Issue name")
    issue_details: str = Field(description="potential issues due to fault injection")
    manifests: List[str] = Field(description="manifest names having the issues")
    problematic_config: str = Field(description="problematic configuration causing the issues (no need to suggest improvements).")

class K8sIssues(BaseModel):
    issues: List[K8sIssue] = Field(description="List issues with its name, potential issues due to fault injection, and manifest configuration causing the issues (no need to suggest improvements).")


#------------------
# agent definition
#------------------
class K8sWeaknessSummaryAgent:
    def __init__(
        self,
        llm: LLM,
        message_logger: MessageLogger
    ) -> None:
        self.llm = llm
        self.message_logger = message_logger
        self.agent = build_json_agent(
            llm=llm,
            chat_messages=[
                ("system", SYS_SUMMARIZE_K8S_WEAKNESSES),
                ("human", USER_SUMMARIZE_K8S_WEAKNESSES)
            ],
            pydantic_object=K8sIssues,
            is_async=False
        )

    def summarize_weaknesses(self, k8s_yamls: List[File]) -> Tuple[LLMLog, str]:
        self.logger = LoggingCallback(name="k8s_summary", llm=self.llm)
        debouncer = StreamDebouncer()
        placeholder = self.message_logger.placeholder()
        
        final_output = None
        for output in self.agent.stream(
            {"k8s_yamls": self.get_k8s_yamls_str(k8s_yamls)},
            {"callbacks": [self.logger]}
        ):
            final_output = output  # Keep track of the final output
            if debouncer.should_update():
                placeholder.write(self.get_text(output))
        
        # Use the final output for the final text generation
        final_text = self.get_text(final_output) if final_output is not None else "No weakness summary generated."
        placeholder.write(final_text)
        return self.logger.log, final_text
    
    def get_text(self, output: Union[dict, str, Any]) -> str:
        """
        Safely extract text from the LLM response, handling different response formats from various providers.
        """
        if output is None:
            return "No response received from the LLM."
        
        text = ""
        
        # Try to extract issues using safe response handling
        issues = safe_get_response_field(output, "issues", None)
        
        # Handle case where issues might be None or not a list
        if issues is None:
            return "No weakness issues found in the response."
        
        # Handle case where issues is not a list
        if not isinstance(issues, list):
            # Try to parse if it's a string representation
            if isinstance(issues, str):
                try:
                    import json
                    parsed_issues = json.loads(issues)
                    if isinstance(parsed_issues, list):
                        issues = parsed_issues
                    else:
                        return f"Unexpected issues format: {str(issues)[:200]}..."
                except (json.JSONDecodeError, ImportError):
                    return f"Unexpected response format for issues: {str(issues)[:200]}..."
            else:
                return f"Unexpected response format for issues: {str(issues)[:200]}..."
        
        # Process each issue safely
        for i, issue in enumerate(issues):
            if not isinstance(issue, dict):
                text += f"Issue #{i}: Invalid issue format\n"
                continue
                
            # Safely extract each field from the issue
            name = safe_get_response_field(issue, "issue_name")
            details = safe_get_response_field(issue, "issue_details")
            manifests = safe_get_response_field(issue, "manifests")
            config = safe_get_response_field(issue, "problematic_config")
            
            if name:
                text += f"Issue #{i}: {name}\n"
            if details:
                text += f"  - details: {details}\n"
            if manifests:
                text += f"  - manifests having the issues: {manifests}\n"
            if config:
                text += f"  - problematic config: {config}\n\n"
        
        return text if text.strip() else "No weakness issues extracted from the response."

    def get_k8s_yamls_str(self, k8s_yamls: List[File]) -> str:
        input_str = ""
        for k8s_yaml in k8s_yamls:
            input_str += f"```{k8s_yaml.fname}```\n```yaml\n{k8s_yaml.content}\n```\n\n"
        return input_str