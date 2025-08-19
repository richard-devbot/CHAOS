from typing import List, Dict, Any

import streamlit as st

from .functions import limit_string_length, MessageLogger


# stand-alone spiner in streamlit
# ref: https://github.com/streamlit/streamlit/issues/6799
class Spinner:
    def __init__(self, text = "In progress..."):
        self.text = text
        self.empty = st.empty()
        self._spinner = iter(self._start()) # This creates an infinite spinner
        next(self._spinner) #  This starts it
        
    def _start(self):
        with st.spinner(self.text):
            yield
    
    def end(self, text: str = None): # This ends it
        next(self._spinner, None)
        if text is not None:
            self.empty.write(text)

class StreamlitPlaceholder:
    def __init__(self):
        self.placeholder = st.empty()
        self.type = ""
        self.content = ""

    def write(self, text: str) -> None:
        self.placeholder.write(text)
        self.type = "write"
        self.content = text

    def code(self, text: str, language: str = None) -> None:
        self.placeholder.code(text, language=language)
        self.type = "code"
        self.content = text
        self.language = language

    def expander(self, text: str, expanded: bool = True):
        self.type = "expander"
        self.content = StreamlitExpander(text, expanded, self.placeholder)
        return self.content

class StreamlitContainer:
    def __init__(self, border: bool = False) -> None:
        self.container = st.container(border=border)
        self.type = "container"
        self.border = border
        self.children = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def write(self, text: str) -> None:
        self.children.append({"type": "write", "content": text})
        self.container.write(text)

    def code(self, code: str, language: str = None) -> None:
        self.children.append({"type": "code", "content": code, "language": language})
        self.container.code(code, language=language)

    def placeholder(self) -> StreamlitPlaceholder:
        with self.container:
            placeholder = StreamlitPlaceholder()
        self.children.append({
            "type": "placeholder",
            "content": placeholder
        })
        return placeholder

    # def expander(self, text: str, expanded: bool = True):
    #     exp = self.container.expander(text, expanded=expanded)
    #     self.children.append({"type": "expander", "title": text, "expanded": expanded})
    #     return exp

class StreamlitExpander:
    def __init__(
        self,
        text: str,
        expanded: bool = True,
        parent_placeholder: StreamlitPlaceholder = None
    ):
        if parent_placeholder is None:
            self.expander = st.expander(text, expanded=expanded)
        else:
            self.expander = parent_placeholder.expander(text, expanded=expanded)
        self.text = text
        self.expanded = expanded
        self.children = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def write(self, text: str) -> None:
        self.children.append({"type": "write", "content": text})
        self.expander.write(text)

    def code(self, code: str, language: str = None) -> None:
        self.children.append({"type": "code", "content": code, "language": language})
        self.expander.code(code, language=language)

    def container(self, border: bool = False) -> StreamlitContainer:
        with self.expander:
            cont = StreamlitContainer(border=border)
        self.children.append({
            "type": "container",
            "border": border,
            "content": cont
        })
        return cont

    def placeholder(self) -> StreamlitPlaceholder:
        with self.expander:
            placeholder = StreamlitPlaceholder()
        self.children.append({
            "type": "placeholder",
            "content": placeholder
        })
        return placeholder

class StreamlitLogger(MessageLogger):
    def write(self, text: str) -> None:
        self.messages.append({
            "type": "write",
            "content": text
        })
        st.write(text)

    def code(
        self,
        code: str,
        language: str = None
    ) -> None:
        self.messages.append({
            "type": "code",
            "content": code,
            "language": language
        })
        st.code(code, language=language)

    def placeholder(self) -> StreamlitPlaceholder:
        placeholder = StreamlitPlaceholder()
        self.messages.append({
            "type": "placeholder",
            "content": placeholder
        })
        return placeholder
    
    def subheader(
        self,
        text: str,
        divider: str
    ) -> None:
        self.messages.append({
            "type": "subheader",
            "content": text,
            "divider": divider
        })
        st.subheader(text, divider=divider)

    def container(self, border: bool = False) -> StreamlitContainer:
        cont = StreamlitContainer(border=border)
        self.messages.append({
            "type": "container",
            "border": border,
            "content": cont
        })
        return cont
    
    def expander(self, text: str, expanded: bool = True) -> StreamlitExpander:
        expander = StreamlitExpander(text, expanded)
        self.messages.append({
            "type": "expander",
            "content": expander
        })
        return expander
    
    def iframe(
        self,
        url: str,
        height: int,
        scrolling: bool
    ) -> None:
        st.components.v1.iframe(url, height=height, scrolling=scrolling)
        self.messages.append({
            "type": "iframe",
            "content": url,
            "height": height,
            "scrolling": scrolling
        })

    def display_history(self) -> None:
        self._render_messages(self.messages)

    def _render_messages(self, messages, parent=None) -> None:
        for message in messages:
            t = message["type"]
            if t == "write":
                st.write(message["content"])
            elif t == "code":
                st.code(message["content"], message["language"])
            elif t == "subheader":
                st.subheader(message["content"], divider=message["divider"])
            elif t == "iframe":
                st.components.v1.iframe(
                    message["content"],
                    height=message["height"],
                    scrolling=message["scrolling"]
                )
            elif t == "container":
                with st.container(border=message["border"]):
                    self._render_messages(message["content"].children)
            elif t == "expander":
                with st.expander(message["content"].text, expanded=message["content"].expanded):
                    self._render_messages(message["content"].children)
            elif t == "placeholder":
                if hasattr(message["content"], "type") and message["content"].type == "write":
                    st.write(message["content"].content)
                elif hasattr(message["content"], "type") and message["content"].type == "code":
                    st.code(message["content"].content, language=message["content"].language)
                elif hasattr(message["content"], "type") and message["content"].type == "expander":
                    with st.expander(message["content"].content.text, expanded=message["content"].content.expanded):
                        self._render_messages(message["content"].content.children)

class StreamlitDisplayHandler:
    """Display handler implementation for Streamlit UI"""
    
    def __init__(
        self,
        message_logger: StreamlitLogger,
        header: str = ""
    ):
        self.message_logger = message_logger
        # Create empty containers for dynamic content updates
        self.message_logger.write(header)
        self.cmd_container = self.message_logger.placeholder()
        self.idx = -1
        self.cmd = []
        self.output_text = []

    def on_start(self, cmd: str = ""):
        """Initialize display with progress status"""
        self.cmd.append(limit_string_length(cmd))
        self.output_text.append("")
        self.idx += 1
        output_text = ""
        for i in range(len(self.cmd)):
            if i < len(self.cmd) - 1:
                output_text += f"$ {self.cmd[i]}\n{self.output_text[i]}"
            else:
                output_text += f"$ {self.cmd[i]}"
        self.cmd_container.code(output_text, language="powershell")
        return self.cmd_container

    def on_output(self, output: str):
        """Update output container with new content"""
        if output != "":
            self.output_text[self.idx] += output
            self.output_text[self.idx] = limit_string_length(self.output_text[self.idx])
        output_text = ""
        for i in range(len(self.cmd)):
            output_text += f"$ {self.cmd[i]}\n{self.output_text[i]}"
        self.cmd_container.code(output_text, language="powershell")

    def on_success(self, output: str = ""):
        """Update status to show successful completion"""
        if output != "":
            output_text = ""
            for i in range(self.idx):
                output_text += f"$ {self.cmd[i]}\n{self.output_text[i]}"
            self.output_text[self.idx] = limit_string_length(output)
            output_text += f"$ {self.cmd[self.idx]}\n{self.output_text[self.idx]}"
            self.cmd_container.code(output_text, language="powershell")

    def on_error(self, error: str):
        """Update status and output to show error details"""
        output_text_tmp = f"Error: {error}"
        self.output_text[self.idx] += limit_string_length(output_text_tmp)
        output_text = ""
        for i in range(len(self.cmd)):
            output_text += f"$ {self.cmd[i]}\n{self.output_text[i]}"
        self.cmd_container.code(output_text, language="powershell")


class StreamlitDisplayContainer:
    def __init__(
        self,
        message_logger: StreamlitLogger,
        text: str = "##### ",
        expanded: bool = True
    ) -> None:
        self.message_logger = message_logger
        self.header_empty = self.message_logger.placeholder()
        self.header = self.header_empty.expander(text, expanded=expanded)
        self.subcontainers = []
        self.subsubcontainers = []

    def update_header(
        self,
        text: str,
        expanded: bool = True
    ) -> None:
        old = self.header
        new = self.header_empty.expander(text, expanded=expanded)
        new.children = old.children

    def complete_header(self):
        pass

    def create_subcontainer(
        self,
        id: str,
        border: bool = True,
        header: str = ""
    ):
        subcontainer = self.header.container(border=border)
        self.subcontainers.append({"id": id, "item": subcontainer})
        if header != "":
            subcontainer.write(header)

    def create_subsubcontainer(
        self,
        subcontainer_id: str,
        subsubcontainer_id: str,
        text: str = None,
        is_code: bool = False,
        language: str = "python"
    ) -> None:
        subcontainer = self.get_item_from_id(self.subcontainers, subcontainer_id)
        try:
            self.get_item_from_id(self.subsubcontainers, subsubcontainer_id)
            raise RuntimeError(f"The subsub container with id '{subsubcontainer_id}' already exists. No duplicated ids are allowed.")
        except RuntimeError:
            placeholder = subcontainer.placeholder()
            self.subsubcontainers.append({"id": subsubcontainer_id, "item": placeholder})
            if text is not None:
                self.update_subsubcontainer(text, subsubcontainer_id, is_code, language)

    def update_subsubcontainer(
        self,
        text: str,
        id: str,
        is_code: bool = False,
        language: str = "python"
    ) -> None:
        subsubcontainer = self.get_item_from_id(self.subsubcontainers, id)
        if is_code:
            subsubcontainer.code(text, language=language)
        else:
            subsubcontainer.write(text)
    
    def get_item_from_id(
        self,
        data: List[Dict[str, Any]],
        id: str
    ):
        if (item := next((item["item"] for item in data if item["id"] == id), None)) is not None:
            return item
        raise RuntimeError(f"Could not find an sub(sub)container with id '{id}' in the dataset.")

    def get_subcontainer(self, id: str):
        return self.get_item_from_id(self.subcontainers, id)

    def get_subsubcontainer(self, id: str):
        return self.get_item_from_id(self.subsubcontainers, id)