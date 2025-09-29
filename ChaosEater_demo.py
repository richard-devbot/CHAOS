import os
import os
import time
import yaml
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime
import json

import streamlit as st
import redis
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from streamlit_extras.bottom_container import bottom

import chaos_eater.utils.app_utils as app_utils
from chaos_eater.chaos_eater import ChaosEater, ChaosEaterInput
from chaos_eater.ce_tools.ce_tool import CEToolType, CETool
from chaos_eater.utils.llms import load_llm, check_existing_key, get_env_key_name
from chaos_eater.utils.functions import get_timestamp, type_cmd, is_binary
from chaos_eater.utils.k8s import remove_all_resources_by_namespace
from chaos_eater.utils.schemas import File
from chaos_eater.utils.constants import CHAOSEATER_IMAGE_PATH, CHAOSEATER_LOGO_PATH, CHAOSEATER_ICON, CHAOSEATER_IMAGE
from chaos_eater.utils.exceptions import ModelNotFoundError
from chaos_eater.utils.streamlit import StreamlitLogger, StreamlitUsageDisplayCallback, StreamlitInterruptCallback


# for debug
from langchain.globals import set_verbose
import langchain
langchain.debug = True
set_verbose(True)

WORK_DIR  = "sandbox"
NAMESPACE = "chaos-eater"
EXAMPLE_DIR = "./examples"


REQUEST_URL_INSTRUCTIONS = """
- When using k6 in steady-state definition, always select a request URL from the following options (other requests are invalid):
  1. http://front-end.sock-shop.svc.cluster.local/
  2. http://front-end.sock-shop.svc.cluster.local/catalogue?size=10
  3. http://front-end.sock-shop.svc.cluster.local/detail.html?id=<ID>
  Replace <ID> with an available ID: [`03fef6ac-1896-4ce8-bd69-b798f85c6e0b`, `3395a43e-2d88-40de-b95f-e00e1502085b`, `510a0d7e-8e83-4193-b483-e27e09ddc34d`, `808a2de1-1aaa-4c25-a9b9-6612e8f29a38`, `819e1fbf-8b7e-4f6d-811f-693534916a8b`, `837ab141-399e-4c1f-9abc-bace40296bac`, `a0a4f044-b040-410d-8ead-4de0446aec7e`, `d3588630-ad8e-49df-bbd7-3167f7efb246`, `zzz4f044-b040-410d-8ead-4de0446aec7e`]
  4. http://front-end.sock-shop.svc.cluster.local/category/
  5. http://front-end.sock-shop.svc.cluster.local/category?tags=<TAG>
  Replace <TAG> with an available tag: [`magic`, `action`, `blue`, `brown`, `black`, `sport`, `formal`, `red`, `green`, `skin`, `geek`]
  6. http://front-end.sock-shop.svc.cluster.local/basket.html"""


@st.experimental_dialog("Confirm Pulling Model")
def pull_model(model_name: str):
    st.write(f"The specified model: {model_name} was not found in the available model list. Do you want to pull it?")
    if st.button("Pull the model"):
        print(f"{model_name}")
        subprocess.run(
            ["docker", "exec", "-it", "ollama", "ollama", "pull", model_name.split("ollama/", 1)[1]],
            check=True
        )
        st.rerun()

def init_choaseater(
    model_name: str = "openai/gpt-4o",
    temperature: float = 0.0,
    port: int = 8000,
    seed: int = 42,
    github_base_url: str = "https://models.github.ai/inference"
) -> None:
    provider = model_name.split("/")[0]
    if provider in ["openai", "anthropic", "google", "github"]:
        if provider == "github":
            if st.session_state.github_token != "":
                os.environ["GITHUB_TOKEN"] = st.session_state.github_token
            if st.session_state.github_base_url != "":
                github_base_url = st.session_state.github_base_url
        else:
            if st.session_state.api_key != "":
                os.environ[get_env_key_name(provider)] = st.session_state.api_key
    elif provider == "bedrock":
        # Set AWS env vars from session_state if provided
        if hasattr(st.session_state, 'aws_access_key_id') and st.session_state.aws_access_key_id != "":
            os.environ["AWS_ACCESS_KEY_ID"] = st.session_state.aws_access_key_id
        if hasattr(st.session_state, 'aws_secret_access_key') and st.session_state.aws_secret_access_key != "":
            os.environ["AWS_SECRET_ACCESS_KEY"] = st.session_state.aws_secret_access_key
        if hasattr(st.session_state, 'aws_session_token') and st.session_state.aws_session_token != "":
            os.environ["AWS_SESSION_TOKEN"] = st.session_state.aws_session_token
        if hasattr(st.session_state, 'aws_region') and st.session_state.aws_region != "":
            os.environ["AWS_REGION"] = st.session_state.aws_region
        # Validate if keys are set
        if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            st.error("AWS Bedrock requires Access Key ID and Secret Access Key. Please provide them in the sidebar.")
            return
    try:
        llm = load_llm(
            model_name=model_name,
            temperature=temperature,
            port=port,
            seed=seed,
            github_base_url=github_base_url
        )
    except ModelNotFoundError:
        pull_model(model_name)
        llm = load_llm(
            model_name=model_name,
            temperature=temperature,
            port=port,
            seed=seed,
            github_base_url=github_base_url
        )

    st.session_state.chaoseater = ChaosEater(
        llm=llm,
        message_logger=st.session_state.message_logger,
        ce_tool=CETool.init(CEToolType.chaosmesh),
        work_dir=WORK_DIR,
        namespace=NAMESPACE
    )
    st.session_state.model_name = model_name
    st.session_state.temperature = temperature
    st.session_state.seed = seed

def safe_yaml_load(content: str) -> dict:
    """
    Safely load YAML content, handling both single and multiple documents.
    For multiple documents, returns the first document.
    """
    try:
        # First try single document parsing
        result = yaml.safe_load(content)
        if result is None:
            # Handle empty YAML files
            documents = list(yaml.safe_load_all(content))
            if documents and documents[0] is not None:
                return documents[0]
            else:
                raise ValueError("No valid YAML documents found")
        return result
    except yaml.YAMLError:
        # If that fails, try multi-document parsing and use the first document
        try:
            documents = list(yaml.safe_load_all(content))
            if documents and documents[0] is not None:
                return documents[0]  # Use the first document
            else:
                raise ValueError("No valid YAML documents found")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML content: {str(e)}")

def get_experiment_details(cycle_dir: str) -> dict:
    """
    Extract experiment details from a cycle directory
    """
    details = {
        "status": "Unknown",
        "duration": "N/A",
        "timestamp": "N/A",
        "model": "N/A",
        "experiments_count": 0,
        "success": False
    }
    
    try:
        # Check if output.json exists
        output_path = os.path.join(cycle_dir, "outputs", "output.json")
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                data = json.load(f)
                
            # Extract basic info
            if "model_name" in data:
                details["model"] = data["model_name"]
            
            # Check experiment results
            if "ce_cycle" in data and "result_history" in data["ce_cycle"]:
                results = data["ce_cycle"]["result_history"]
                details["experiments_count"] = len(results)
                if results:
                    details["success"] = results[-1].get("all_tests_passed", False)
                    details["status"] = "Success" if details["success"] else "Failed"
            
            # Calculate total duration
            if "run_time" in data:
                total_time = 0
                for phase_times in data["run_time"].values():
                    if isinstance(phase_times, list):
                        total_time += sum(phase_times)
                    elif isinstance(phase_times, (int, float)):
                        total_time += phase_times
                details["duration"] = f"{total_time:.1f}s"
                        
        # Get timestamp from directory name or file modification time
        if cycle_dir.startswith("cycle_"):
            timestamp_str = cycle_dir[6:]  # Remove "cycle_" prefix
            try:
                # Parse timestamp format: YYYYMMDD_HHMMSS
                dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                details["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                # Fallback to file modification time
                stat = os.stat(cycle_dir)
                details["timestamp"] = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error getting experiment details for {cycle_dir}: {e}")
    
    return details

def find_message_logs(root: str = "."):
    logs = {}
    for entry in os.listdir(root):
        cycle_path = os.path.join(root, entry)
        if os.path.isdir(cycle_path):
            target = os.path.join(cycle_path, "outputs", "message_log.pkl")
            if os.path.exists(target):
                logs[entry] = target
    return logs

def main():
    #---------------------------
    # initialize session states
    #---------------------------
    if "state_list" not in st.session_state:
        st.session_state.state_list = {}
    if "session_id" not in st.session_state:
        session_ctx = get_script_run_ctx()
        st.session_state.session_id = session_ctx.session_id
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "count" not in st.session_state:
        st.session_state.count = 0
    if "is_first_run" not in st.session_state:
        st.session_state.is_first_run = True
    if "input" not in st.session_state:
        st.session_state.input = None
    if "submit" not in st.session_state:
        st.session_state.submit = False
    if "model_name" not in st.session_state:
        st.session_state.model_name = "openai/gpt-4o-2024-08-06"
    if "seed" not in st.session_state:
        st.session_state.seed = 42
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.0
    if "message_logger" not in st.session_state:
        st.session_state.message_logger = StreamlitLogger()
    if "selected_cycle" not in st.session_state:
        st.session_state.selected_cycle = ""
    if "github_token" not in st.session_state:
        st.session_state.github_token = ""
    if "github_base_url" not in st.session_state:
        st.session_state.github_base_url = "https://models.github.ai/inference"

    #--------------
    # CSS settings
    #--------------
    st.set_page_config(
        page_title="ChaosEater",
        page_icon=CHAOSEATER_IMAGE,
        # layout="wide"
    )
    app_utils.apply_hide_st_style()
    app_utils.apply_hide_height0_components()
    app_utils.apply_centerize_components_vertically()
    app_utils.apply_remove_sidebar_topspace()
    app_utils.apply_enable_auto_scroll()
    
    #---------
    # sidebar
    #---------
    st.logo(CHAOSEATER_LOGO_PATH)
    with st.sidebar:
        #----------
        # settings
        #----------
        stop_button = st.empty()
        with st.expander("General settings", expanded=True):
            #-----------------
            # model selection
            #-----------------
            selected_model = st.selectbox(
                "Model",
                (
                    "openai/gpt-4o-2024-08-06",
                    "google/gemini-2.0-flash-lite",
                    "anthropic/claude-3-5-sonnet-20241022",
                    "bedrock/anthropic.claude-3-5-sonnet-20241022",
                    "github/gpt-4o",
                    "github/gpt-4o-mini",
                    "ollama/qwen3:32b",
                    "custom"
                )
            )
            if selected_model == "custom":
                model_name = st.text_input("Enter custom model name", placeholder="ollama/gpt-oss:120b")
                if model_name is None:
                    model_name = "openai/gpt-4o-2024-08-06"
            else:
                model_name = selected_model
            
            # Handle different provider authentication
            if model_name.startswith("bedrock/"):
                provider = "bedrock"
                # For Bedrock, check if keys are set in session_state or env
                has_valid_key = check_existing_key(provider)
                with st.expander("AWS Bedrock Credentials", expanded=not has_valid_key):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_input(
                            label="AWS Access Key ID",
                            key="aws_access_key_id",
                            placeholder="Enter your AWS Access Key ID",
                            type="password",
                            help="Required for Bedrock access"
                        )
                    with col2:
                        st.text_input(
                            label="AWS Secret Access Key",
                            key="aws_secret_access_key",
                            placeholder="Enter your AWS Secret Access Key",
                            type="password",
                            help="Required for Bedrock access"
                        )
                    col3, col4 = st.columns(2)
                    with col3:
                        session_token = st.text_input(
                            label="AWS Session Token (optional)",
                            key="aws_session_token",
                            placeholder="Enter if using temporary credentials",
                            type="password",
                            help="Optional for temporary credentials"
                        )
                    with col4:
                        region = st.text_input(
                            label="AWS Region",
                            key="aws_region",
                            value="us-east-1",
                            placeholder="us-east-1",
                            help="Default: us-east-1"
                        )
                if has_valid_key:
                    st.info("Valid AWS Bedrock credentials detected")
            elif model_name.startswith("github/"):
                provider = "github"
                has_valid_key = check_existing_key(provider)
                if has_valid_key:
                    help_text = "A valid GitHub token is already set"
                else:
                    help_text = "Enter your GitHub token"
                st.text_input(
                    label="GitHub Token",
                    key="github_token",
                    placeholder=help_text,
                    type="password"
                )
                st.text_input(
                    label="GitHub Base URL",
                    key="github_base_url",
                    value=st.session_state.github_base_url,
                    placeholder="https://models.github.ai/inference"
                )
            elif model_name.startswith(("openai", "google", "anthropic")):
                provider = model_name.split("/")[0]
                has_valid_key = check_existing_key(provider)
                if has_valid_key:
                    help_text = f"A valid {provider.capitalize()} API key is already set"
                else:
                    help_text = f"Enter your {provider.capitalize()} API key"
                st.text_input(
                    label="API key",
                    key="api_key",
                    placeholder=help_text,
                    type="password"
                )
            
            #-------------------
            # cluster selection
            #-------------------
            avail_cluster_list = app_utils.get_available_clusters()
            FULL_CAP_MSG = "No clusters available right now. Please wait until a cluster becomes available."
            if len(avail_cluster_list) == 0:
                avail_cluster_list = (FULL_CAP_MSG,)
            cluster_name = st.selectbox(
                "Cluster selection",
                avail_cluster_list,
                key="cluster_name"
            )
            app_utils.monitor_session(st.session_state.session_id)
            st.button(
                "Clean the cluster",
                key="clean_k8s",
                on_click=remove_all_resources_by_namespace,
                args=(cluster_name, NAMESPACE, )
            )

            #--------------------
            # parameter settings
            #--------------------
            clean_cluster_before_run = st.checkbox("Clean the cluster before run", value=True)
            clean_cluster_after_run = st.checkbox("Clean the cluster after run", value=True)
            is_new_deployment = st.checkbox("New deployment", value=True)
            seed = st.number_input("Seed for LLMs", 42)
            temperature = st.number_input("Temperature for LLMs", 0.0)
            max_num_steadystates = st.number_input("Max. number of steady states", 3)
            max_retries = st.number_input("Max retries", 3)

        #-------------------
        # history of cycles
        #-------------------
        clicked_cycle = None
        with st.expander("📊 Recent Experiments", expanded=True):
            logs = find_message_logs(WORK_DIR)
            if not logs:
                st.info("🔬 No experiments found. Run your first chaos engineering cycle!")
            else:
                sorted_logs = sorted(
                    logs.items(),
                    key=lambda x: os.path.getmtime(x[1]),
                    reverse=True
                )
                
                for i, (name, path) in enumerate(sorted_logs[:10]):  # Show only last 10 experiments
                    cycle_dir = os.path.dirname(os.path.dirname(path))  # Go up from outputs/message_log.pkl
                    details = get_experiment_details(cycle_dir)
                    
                    # Create a more informative button label
                    status_emoji = "✅" if details["success"] else "❌" if details["status"] == "Failed" else "⏳"
                    button_label = f"{status_emoji} {name}"
                    
                    # Show experiment details in an expander for the first few experiments
                    if i < 3:  # Show details for the 3 most recent
                        with st.container():
                            if st.button(
                                button_label,
                                key=f"cycle_{name}",
                                use_container_width=True,
                                type="primary" if st.session_state.selected_cycle == name else "secondary"
                            ):
                                clicked_cycle = (name, path)
                            
                            # Show experiment summary
                            col1, col2 = st.columns(2)
                            with col1:
                                st.caption(f"🤖 {details['model']}")
                                st.caption(f"⏱️ {details['duration']}")
                            with col2:
                                st.caption(f"🧪 {details['experiments_count']} experiments")
                                st.caption(f"📅 {details['timestamp'][:10]}")
                            
                            st.divider()
                    else:
                        # For older experiments, just show the button
                        if st.button(
                            button_label,
                            key=f"cycle_{name}",
                            use_container_width=True,
                            type="primary" if st.session_state.selected_cycle == name else "secondary"
                        ):
                            clicked_cycle = (name, path)

    if clicked_cycle:
        name, path = clicked_cycle
        st.session_state.message_logger = StreamlitLogger.load(path)
        st.session_state.usage_displayer.load(Path(path).with_name("output.json"))
        st.session_state.selected_cycle = name
        st.session_state.is_first_run = False
        st.rerun()

    with st.sidebar:
        #---------------------------
        # usage: tokens and billing
        #---------------------------
        with st.expander("Usage", expanded=True):
            st.session_state.usage_displayer = StreamlitUsageDisplayCallback(model_name)
        
        #--------------------------
        # experiment monitoring
        #--------------------------
        with st.expander("🔍 Experiment Monitoring", expanded=False):
            st.markdown("**Monitoring Options:**")
            
            # Check if we're in a containerized environment (common for AWS EC2)
            if os.path.exists("/.dockerenv") or os.environ.get("container"):
                st.info("🌍 **Running in container/EC2**")
                st.markdown("To access experiment monitoring:")
                st.code("http://<YOUR_EC2_IP>:2333", language="text")
                st.caption("⚠️ Make sure port 2333 is open in your EC2 security group")
            else:
                st.info("🖥️ **Local environment**")
                st.markdown("Experiment monitoring available at:")
                st.code("http://localhost:2333", language="text")
                
            # Show recent experiment files location
            st.markdown("**📁 Results Storage:**")
            st.code(f"{WORK_DIR}/cycle_YYYYMMDD_HHMMSS/", language="text")
            
            if st.button("📂 Open Results Folder", use_container_width=True):
                if os.path.exists(WORK_DIR):
                    # For demonstration, show the path
                    st.success(f"Results stored in: {os.path.abspath(WORK_DIR)}")
                else:
                    st.warning("No experiments found yet")
        
        #-----------------
        # command history
        #-----------------
        if not st.session_state.is_first_run:
            st.write("Command history")


    #------------------------
    # initialize chaos eater
    #------------------------
    # initialization
    if (
        "chaoseater" not in st.session_state
        or model_name != st.session_state.model_name
        or seed != st.session_state.seed
        or temperature != st.session_state.temperature
    ):
        init_choaseater(
            model_name=model_name,
            seed=seed,
            temperature=temperature
        )

    # greeding 
    if len(st.session_state.chat_history) == 0 and st.session_state.is_first_run:
        app_utils.add_chaoseater_icon(CHAOSEATER_IMAGE_PATH)
        if st.session_state.count == 0: # streaming
            greeding = "Let's dive into Chaos together :)"
            elem = st.empty()
            words = ""
            for word in list(greeding):
                if word == "C":
                    words += "<font color='#7fff00'>" + word
                elif word == "s":
                    words += word + "</font>"
                else:
                    words += word
                elem.markdown(f'<center> <h3> {words} </h3> </center>', unsafe_allow_html=True)
                time.sleep(0.06)
        else:
            greeding = "Let's dive into <font color='#7fff00'>Chaos</font> together :)"
            st.markdown(f'<center> <h3> {greeding} </h3> </center>', unsafe_allow_html=True)

    #----------
    # examples
    #----------
    def submit_example(number: int, example_dir: str, instructions: str) -> None:
        decorated_func = st.experimental_dialog(f"Example input #{number}")(submit_example_internal)
        decorated_func(example_dir, instructions)

    def submit_example_internal(example_dir: str, instructions: str) -> None:
        # load the project
        skaffold_yaml = None
        project_files_tmp = []
        for root, _, files in os.walk(example_dir):
            for entry in files:
                fpath = os.path.join(root, entry)
                if os.path.isfile(fpath):
                    with open(fpath, "rb") as f:
                        file_content = f.read()
                    if is_binary(file_content):
                        content = file_content
                    else:
                        content = file_content.decode('utf-8')
                    if os.path.basename(fpath) == "skaffold.yaml":
                        skaffold_yaml = File(
                            path=fpath,
                            content=content,
                            work_dir=EXAMPLE_DIR,
                            fname=fpath.removeprefix(f"{EXAMPLE_DIR}/")
                        )
                    else:
                        project_files_tmp.append(File(
                                path=fpath,
                                content=content,
                                work_dir=EXAMPLE_DIR,
                                fname=fpath.removeprefix(f"{EXAMPLE_DIR}/")
                        ))
        input_tmp = ChaosEaterInput(
            skaffold_yaml=skaffold_yaml,
            files=project_files_tmp,
            ce_instructions=instructions
        )
        if skaffold_yaml is None:
            st.error("Error parsing skaffold.yaml: skaffold_yaml is None")
            return
            
        skaffold_yaml_dir = os.path.dirname(skaffold_yaml.path or "")
        k8s_yamls_tmp = []
        try:
            skaffold_config = safe_yaml_load(str(skaffold_yaml.content))
            raw_yaml_paths = skaffold_config.get("manifests", {}).get("rawYaml", [])
        except ValueError as e:
            st.error(f"Error parsing skaffold.yaml: {e}")
            return
            
        for k8s_yaml_fname in raw_yaml_paths:
            for file in project_files_tmp:
                if f"{skaffold_yaml_dir}/{k8s_yaml_fname}" == file.path:
                    k8s_yamls_tmp.append(File(
                        path=file.path,
                        content=file.content,
                        fname=file.fname
                    ))
        # display the example
        st.write("### Input K8s manifest(s):")
        for k8s_yaml in k8s_yamls_tmp:
            st.write(f"```{k8s_yaml.fname}```")
            st.code(k8s_yaml.content)
        st.write("### Instructions:")
        st.write(instructions)
        with st.columns(3)[1]:
            # submit the example
            if st.button("Try this one"):
                st.session_state.input = input_tmp
                st.session_state.submit = True
                st.rerun()

    if st.session_state.is_first_run:
        app_utils.apply_remove_example_bottomspace(px=0)
        st.session_state.bottom_container = bottom()
        with st.session_state.bottom_container:
            #----------
            # examples
            #----------
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(
                    "example#1:  \nNginx w/ detailed CE instructions",
                    key="example1",
                    use_container_width=True
                ):
                    submit_example(
                        number=1,
                        example_dir=f"{EXAMPLE_DIR}/nginx",
                        instructions="- The Chaos-Engineering experiment must be completed within 1 minute.\n- List ONLY one steady state about Pod Count.\n- Conduct pod-kill"
                    )
            with col2:
                if st.button(
                    "example#2:  \nNginx w/ limited experiment duration",
                    key="example2",
                    use_container_width=True
                ):
                    submit_example(
                        number=2,
                        example_dir=f"{EXAMPLE_DIR}/nginx",
                        instructions="The Chaos-Engineering experiment must be completed within 1 minute."
                    )
            with col3:
                if st.button(
                    "example#3:  \nSock shop w/ limited experiment duration",
                    key="example3",
                    use_container_width=True
                ):
                    submit_example(
                        number=3,
                        example_dir=f"{EXAMPLE_DIR}/sock-shop-2",
                        instructions=f"- The Chaos-Engineering experiment must be completed within 1 minute.\n{REQUEST_URL_INSTRUCTIONS}"
                    )
            #----------------
            # file uploading
            #----------------
            st.markdown("📁 **Upload Options:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("✅ **ZIP file**: Complete project with skaffold.yaml")
            with col2: 
                st.markdown("✅ **YAML file**: Individual K8s manifests (auto-generates skaffold.yaml)")
            
            upload_col, submit_col = st.columns([10, 2], vertical_alignment="bottom")
            with upload_col:
                file = st.file_uploader(
                    "upload your project (.zip or .yaml files)",
                    type=["zip", "yaml", "yml"],
                    accept_multiple_files=False,
                    label_visibility="hidden",
                    help="📋 Supported formats:\n• ZIP: Complete project structure\n• YAML: Single or multi-document Kubernetes manifests\n• Multiple YAML documents separated by '---' are supported"
                )
                if file is not None:
                    project_files_tmp = []
                    skaffold_yaml = None
                    
                    try:
                        # Handle different file types
                        if file.name.endswith('.zip'):
                            # Process ZIP file (existing logic with validation)
                            with zipfile.ZipFile(file, "r") as z:
                                for file_info in z.infolist():
                                    # only process files, skip directories
                                    if not file_info.is_dir():
                                        with z.open(file_info) as zip_file:
                                            file_content = zip_file.read()
                                            if is_binary(file_content):
                                                content = file_content
                                            else:
                                                content = file_content.decode('utf-8')
                                            fpath = file_info.filename
                                            
                                            # Validate YAML content if it's a YAML file
                                            if fpath.endswith(('.yaml', '.yml')):
                                                try:
                                                    # Test if it's valid YAML (single or multiple documents)
                                                    content_str = content if isinstance(content, str) else content.decode('utf-8')
                                                    if '---' in content_str:
                                                        docs = list(yaml.safe_load_all(content_str))
                                                        if not docs or all(doc is None for doc in docs):
                                                            st.warning(f"⚠️ YAML file '{fpath}' contains no valid documents")
                                                            continue
                                                    else:
                                                        doc = yaml.safe_load(content_str)
                                                        if doc is None:
                                                            st.warning(f"⚠️ YAML file '{fpath}' is empty or invalid")
                                                            continue
                                                except yaml.YAMLError as e:
                                                    st.error(f"❌ Invalid YAML file '{fpath}': {str(e)}")
                                                    continue
                                            
                                            if os.path.basename(fpath) == "skaffold.yaml":
                                                skaffold_yaml = File(
                                                    path=fpath,
                                                    content=content,
                                                    work_dir=EXAMPLE_DIR,
                                                    fname=fpath.removeprefix(EXAMPLE_DIR).lstrip('/')
                                                )
                                            else:
                                                project_files_tmp.append(File(
                                                        path=fpath,
                                                        content=content,
                                                        work_dir=EXAMPLE_DIR,
                                                        fname=fpath.removeprefix(EXAMPLE_DIR).lstrip('/')
                                                ))
                                                
                        elif file.name.endswith(('.yaml', '.yml')):
                            # Process single YAML file
                            content = file.read().decode('utf-8')
                            
                            # Validate YAML content
                            try:
                                if '---' in content:
                                    docs = list(yaml.safe_load_all(content))
                                    if not docs or all(doc is None for doc in docs):
                                        st.error(f"❌ YAML file '{file.name}' contains no valid documents")
                                        st.stop()
                                    st.info(f"📄 Detected {len([d for d in docs if d is not None])} YAML document(s) in {file.name}")
                                else:
                                    doc = yaml.safe_load(content)
                                    if doc is None:
                                        st.error(f"❌ YAML file '{file.name}' is empty or invalid")
                                        st.stop()
                            except yaml.YAMLError as e:
                                st.error(f"❌ Invalid YAML file '{file.name}': {str(e)}")
                                st.stop()
                            
                            if file.name == "skaffold.yaml":
                                skaffold_yaml = File(
                                    path=file.name,
                                    content=content,
                                    work_dir=EXAMPLE_DIR,
                                    fname=file.name
                                )
                            else:
                                # If it's not skaffold.yaml, create a minimal skaffold.yaml pointing to this file
                                project_files_tmp.append(File(
                                    path=file.name,
                                    content=content,
                                    work_dir=EXAMPLE_DIR,
                                    fname=file.name
                                ))
                                
                                # Create a minimal skaffold.yaml
                                skaffold_content = f"""apiVersion: skaffold/v3
kind: Config
metadata:
  name: uploaded-yaml
manifests:
  rawYaml:
    - {file.name}
"""
                                skaffold_yaml = File(
                                    path="skaffold.yaml",
                                    content=skaffold_content,
                                    work_dir=EXAMPLE_DIR,
                                    fname="skaffold.yaml"
                                )
                                st.info(f"🔧 Auto-generated skaffold.yaml for your {file.name}")
                        
                        if skaffold_yaml is not None:
                            st.session_state.input = ChaosEaterInput(
                                skaffold_yaml=skaffold_yaml,
                                files=project_files_tmp
                            )
                            st.success(f"✅ Successfully uploaded {file.name}!")
                            if len(project_files_tmp) > 0:
                                st.info(f"📦 Loaded {len(project_files_tmp)} additional file(s)")
                        else:
                            st.error("❌ No valid skaffold.yaml found. Please ensure your project contains a skaffold.yaml file.")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {file.name}: {str(e)}")
                        st.error("💡 **Tip**: Ensure your ZIP file contains a valid skaffold.yaml, or upload YAML files directly.")
            with submit_col:
                st.text("")
                if st.button("Submit w/o instructions", key="submit_"):
                    if st.session_state.input != None:
                        st.session_state.input.ce_instructions = ""
                        st.session_state.submit = True
                        st.rerun()
                st.text("")
    else:
        app_utils.apply_remove_example_bottomspace()

    #--------------
    # chat history
    #--------------
    st.session_state.message_logger.display_history()

    #--------------
    # current chat
    #--------------
    if (prompt := st.chat_input(placeholder="Input instructions for your Chaos Engineering", key="chat_input")) or st.session_state.submit:        
        if "chaoseater" in st.session_state and cluster_name != FULL_CAP_MSG:
            if st.session_state.input:
                if st.session_state.is_first_run:
                    st.session_state.is_first_run = False
                    if prompt:
                        st.session_state.input.ce_instructions = prompt
                        st.session_state.submit = True
                    st.rerun()
                input = st.session_state.input
                if prompt:
                    input.ce_instructions = prompt
                st.session_state.input = None
                st.session_state.submit = False
                #-------------
                # user inputs
                #-------------
                with st.chat_message("user"):
                    st.session_state.message_logger.write("##### Your instructions for Chaos Engineering:", role="user")
                    instructions = input.ce_instructions or "No specific instructions provided"
                    st.session_state.message_logger.write(instructions, role="user")
                #---------------------
                # chaoseater response
                #---------------------
                # set the currrent cluster
                if len(avail_cluster_list) > 0 and avail_cluster_list[0] != FULL_CAP_MSG:
                    r = redis.Redis(host='localhost', port=6379, db=0)
                    r.hset("cluster_usage", st.session_state.session_id, cluster_name)
                # display stop button
                if stop_button.button("⏹ Stop", use_container_width=True):
                    st.session_state.stop = True
                with st.chat_message("assistant", avatar=CHAOSEATER_ICON):
                    output = st.session_state.chaoseater.run_ce_cycle(
                        input=input,
                        work_dir=f"{WORK_DIR}/cycle_{get_timestamp()}",
                        kube_context=cluster_name,
                        is_new_deployment=is_new_deployment,
                        clean_cluster_before_run=clean_cluster_before_run,
                        clean_cluster_after_run=clean_cluster_after_run,
                        max_num_steadystates=max_num_steadystates,
                        max_retries=max_retries,
                        callbacks=[
                            st.session_state.usage_displayer,
                            StreamlitInterruptCallback()
                        ]
                    )
                    # download output
                    output_dir = output.work_dir
                    os.makedirs("./temp", exist_ok=True) 
                    zip_path = f"./temp/{os.path.basename(output_dir)}.zip"
                    print(type_cmd(f"zip -r {zip_path} {output_dir}"))
                    with st.columns(3)[1]:
                        with open(zip_path, "rb") as fp:
                            btn = st.download_button(
                                label="Download output (.zip)",
                                data=fp,
                                file_name=f"{os.path.basename(zip_path)}",
                                mime=f"output/zip"
                            )
            else:
                with st.chat_message("assistant", avatar=CHAOSEATER_ICON):
                    st.session_state.message_logger.write("Please input your k8s mainfests!")
        else:
            if cluster_name == FULL_CAP_MSG:
                with st.chat_message("assistant", avatar=CHAOSEATER_ICON):
                    st.session_state.message_logger.write(FULL_CAP_MSG)
            else:
                with st.chat_message("assistant", avatar=CHAOSEATER_ICON):
                    st.session_state.message_logger.write("Please set your API key!")

    st.session_state.count += 1

if __name__ == "__main__":
    main()
