"""
This file defines a useful high-level abstraction to build Gradio chatbots: ChatInterface.
"""

from __future__ import annotations

import builtins
import copy
import dataclasses
import inspect
import os
import warnings
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from functools import wraps
from pathlib import Path
from typing import Any, Literal, Union, cast

import anyio
from gradio_client.documentation import document

from gradio import utils
from gradio.blocks import Blocks
from gradio.components import (
    JSON,
    BrowserState,
    Button,
    Chatbot,
    Component,
    Dataset,
    Markdown,
    MultimodalTextbox,
    State,
    Textbox,
    get_component_instance,
)
from gradio.components.chatbot import (
    ChatMessage,
    ExampleMessage,
    Message,
    MessageDict,
    TupleFormat,
)
from gradio.components.multimodal_textbox import MultimodalPostprocess, MultimodalValue
from gradio.context import get_blocks_context
from gradio.events import Dependency, EditData, SelectData
from gradio.flagging import ChatCSVLogger
from gradio.helpers import create_examples as Examples  # noqa: N812
from gradio.helpers import special_args, update
from gradio.i18n import I18nData
from gradio.layouts import Accordion, Column, Group, Row
from gradio.themes import ThemeClass as Theme

# Add import for PostgreSQL chat history integration
from server.v1.chat_history import (
    get_conversation_history,
    get_all_conversations,
    clear_conversation_history,
)
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import uuid


@document()
class ChatInterface(Blocks):
    def __init__(
        self,
        fn: Callable,
        *,
        type: Literal["messages", "tuples"] | None = None,
        additional_inputs: str | Component | list[str | Component] | None = None,
        additional_inputs_accordion: str | Accordion | None = None,
        additional_outputs: Component | list[Component] | None = None,
        editable: bool = False,
        examples: list[str] | list[MultimodalValue] | list[list] | None = None,
        example_labels: list[str] | None = None,
        example_icons: list[str] | None = None,
        run_examples_on_click: bool = True,
        cache_examples: bool | None = None,
        cache_mode: Literal["eager", "lazy"] | None = None,
        title: str | I18nData | None = None,
        description: str | None = None,
        theme: Theme | str | None = None,
        flagging_mode: Literal["never", "manual"] | None = None,
        flagging_options: list[str] | tuple[str, ...] | None = ("Like", "Dislike"),
        flagging_dir: str = ".gradio/flagged",
        css: str | None = None,
        css_paths: str | Path | Sequence[str | Path] | None = None,
        js: str | Literal[True] | None = None,
        head: str | None = None,
        head_paths: str | Path | Sequence[str | Path] | None = None,
        analytics_enabled: bool | None = None,
        autofocus: bool = True,
        autoscroll: bool = True,
        submit_btn: str | bool | None = True,
        stop_btn: str | bool | None = True,
        concurrency_limit: int | None | Literal["default"] = "default",
        delete_cache: tuple[int, int] | None = None,
        show_progress: Literal["full", "minimal", "hidden"] = "minimal",
        fill_height: bool = True,
        fill_width: bool = False,
        api_name: str | Literal[False] = "chat",
    ):
        super().__init__(
            analytics_enabled=analytics_enabled,
            mode="chat_interface",
            title=title or "Gradio",
            theme=theme,
            css=css,
            css_paths=css_paths,
            js=js,
            head=head,
            head_paths=head_paths,
            fill_height=fill_height,
            fill_width=fill_width,
            delete_cache=delete_cache,
        )
        self.api_name: str | Literal[False] = api_name
        self.type = type
        self.concurrency_limit = concurrency_limit
        if isinstance(fn, ChatInterface):
            self.fn = fn.fn
        else:
            self.fn = fn
        self.is_async = inspect.iscoroutinefunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)
        self.is_generator = inspect.isgeneratorfunction(
            self.fn
        ) or inspect.isasyncgenfunction(self.fn)
        self.examples = examples
        self.examples_messages = self._setup_example_messages(
            examples, example_labels, example_icons
        )
        self.run_examples_on_click = run_examples_on_click
        self.cache_examples = cache_examples
        self.cache_mode = cache_mode
        self.editable = editable
        self.fill_height = fill_height
        self.autoscroll = autoscroll
        self.autofocus = autofocus
        self.title = title
        self.description = description
        self.show_progress = show_progress
        if not type == "messages":
            raise ValueError("history is only supported for type='messages'")

        self.additional_inputs = [
            get_component_instance(i)
            for i in utils.none_or_singleton_to_list(additional_inputs)
        ]
        self.additional_outputs = utils.none_or_singleton_to_list(additional_outputs)
        if additional_inputs_accordion is None:
            self.additional_inputs_accordion_params = {
                "label": "Additional Inputs",
                "open": False,
            }
        elif isinstance(additional_inputs_accordion, str):
            self.additional_inputs_accordion_params = {
                "label": additional_inputs_accordion
            }
        elif isinstance(additional_inputs_accordion, Accordion):
            self.additional_inputs_accordion_params = (
                additional_inputs_accordion.recover_kwargs(
                    additional_inputs_accordion.get_config()
                )
            )
        else:
            raise ValueError(
                f"The `additional_inputs_accordion` parameter must be a string or gr.Accordion, not {builtins.type(additional_inputs_accordion)}"
            )
        self._additional_inputs_in_examples = False
        if self.additional_inputs and self.examples is not None:
            for example in self.examples:
                if not isinstance(example, list):
                    raise ValueError(
                        "Examples must be a list of lists when additional inputs are provided."
                    )
                for idx, example_for_input in enumerate(example):
                    if example_for_input is not None and idx > 0:
                        self._additional_inputs_in_examples = True
                        break
                if self._additional_inputs_in_examples:
                    break

        if flagging_mode is None:
            flagging_mode = os.getenv("GRADIO_CHAT_FLAGGING_MODE", "never")  # type: ignore
        if flagging_mode in ["manual", "never"]:
            self.flagging_mode = flagging_mode
        else:
            raise ValueError(
                "Invalid value for `flagging_mode` parameter."
                "Must be: 'manual' or 'never'."
            )
        self.flagging_options = flagging_options
        self.flagging_dir = flagging_dir

        with self:  # Initialize PostgreSQL-backed conversation storage
            # Load existing conversations from PostgreSQL on startup
            initial_conversations = self._load_conversations_from_postgres()
            self.saved_conversations = State(initial_conversations)
            self.conversation_id = State(None)
            self.saved_input = State()  # Stores the most recent user message
            self.null_component = State()  # Used to discard unneeded values

            with Column():
                self._render_header()
                with Row(scale=1):
                    self._render_history_area()
                    with Column(scale=6):
                        self._render_chatbot_area(submit_btn, stop_btn)
                        self._render_footer()

            self._setup_events()

    def _render_header(self):
        if self.title:
            Markdown(
                f"<h1 style='text-align: center; margin-bottom: 1rem'>{self.title}</h1>"
            )
        if self.description:
            Markdown(self.description)

    def _render_history_area(self):
        with Column(scale=1, min_width=100):
            self.new_chat_button = Button(
                "New chat",
                variant="primary",
                size="md",
                icon=utils.get_icon_path("plus.svg"),
                # scale=0,
            )
            self.chat_history_dataset = Dataset(
                components=[Textbox(visible=False)],
                show_label=False,
                layout="table",
                type="index",
            )

    def _render_chatbot_area(
        self,
        submit_btn: str | bool | None,
        stop_btn: str | bool | None,
    ):
        self.type = self.type or "tuples"
        self.chatbot = Chatbot(
            label="Chatbot",
            scale=1,
            height=400 if self.fill_height else None,
            type=cast(Literal["messages", "tuples"], self.type),
            autoscroll=self.autoscroll,
            examples=(
                self.examples_messages
                if not self._additional_inputs_in_examples
                else None
            ),
        )
        with Group():
            with Row():
                self.textbox = MultimodalTextbox(
                    show_label=False,
                    label="Message",
                    placeholder="Type a message...",
                    scale=7,
                    autofocus=self.autofocus,
                    submit_btn=submit_btn,
                    stop_btn=stop_btn,
                )

        # Hide the stop button at the beginning, and show it with the given value during the generator execution.
        self.original_stop_btn = self.textbox.stop_btn
        self.textbox.stop_btn = False
        self.fake_api_btn = Button("Fake API", visible=False)
        self.api_response = JSON(
            label="Response", visible=False
        )  # Used to store the response from the API call

        # Used internally to store the chatbot value when it differs from the value displayed in the chatbot UI.
        # For example, when a user submits a message, the chatbot UI is immediately updated with the user message,
        # but the chatbot_state value is not updated until the submit_fn is called.
        self.chatbot_state = State(self.chatbot.value if self.chatbot.value else [])

        # Provided so that developers can update the chatbot value from other events outside of `gr.ChatInterface`.
        self.chatbot_value = State(self.chatbot.value if self.chatbot.value else [])

    def _render_footer(self):
        if self.examples:
            self.examples_handler = Examples(
                examples=self.examples,
                inputs=[self.textbox] + self.additional_inputs,
                outputs=self.chatbot,
                fn=self._examples_stream_fn if self.is_generator else self._examples_fn,
                cache_examples=self.cache_examples,
                cache_mode=cast(Literal["eager", "lazy"], self.cache_mode),
                visible=self._additional_inputs_in_examples,
                preprocess=self._additional_inputs_in_examples,
            )

        any_unrendered_inputs = any(
            not inp.is_rendered for inp in self.additional_inputs
        )
        if self.additional_inputs and any_unrendered_inputs:
            with Accordion(**self.additional_inputs_accordion_params):  # type: ignore
                for input_component in self.additional_inputs:
                    if not input_component.is_rendered:
                        input_component.render()

    def _setup_example_messages(
        self,
        examples: list[str] | list[MultimodalValue] | list[list] | None,
        example_labels: list[str] | None = None,
        example_icons: list[str] | None = None,
    ) -> list[ExampleMessage]:
        examples_messages = []
        if examples:
            for index, example in enumerate(examples):
                if isinstance(example, list):
                    example = example[0]
                example_message: ExampleMessage = {}
                if isinstance(example, str):
                    example_message["text"] = example
                elif isinstance(example, dict):
                    example_message["text"] = example.get("text", "")
                    example_message["files"] = example.get("files", [])
                if example_labels:
                    example_message["display_text"] = example_labels[index]
                example_files = example_message.get("files")
                if not example_files:
                    if example_icons:
                        example_message["icon"] = example_icons[index]
                    else:
                        example_message["icon"] = {
                            "path": "",
                            "url": None,
                            "orig_name": None,
                            "mime_type": "text",  # for internal use, not a valid mime type
                            "meta": {"_type": "gradio.FileData"},
                        }
                elif example_icons:
                    example_message["icon"] = example_icons[index]
                examples_messages.append(example_message)
        return examples_messages

    def _generate_chat_title(self, conversation: list[MessageDict]) -> str:
        title = ""
        for message in conversation:
            if message["role"] == "user":
                if isinstance(message["content"], str):
                    title += message["content"]
                    break
                else:
                    title += "📎 "
        if len(title) > 40:
            title = title[:40] + "..."
        return title or "Conversation"

    @staticmethod
    def serialize_components(conversation: list[MessageDict]) -> list[MessageDict]:
        def inner(obj: Any) -> Any:
            if isinstance(obj, list):
                return [inner(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: inner(v) for k, v in obj.items()}
            elif isinstance(obj, Component):
                return obj.value
            return obj

        return inner(conversation)

    def _save_conversation(
        self,
        index: int | None,
        conversation: list[MessageDict],
        saved_conversations: list[list[MessageDict]],
    ):
        if conversation:
            try:
                serialized_conversation = self.serialize_components(conversation)

                # Save to PostgreSQL
                if index is not None and index < len(saved_conversations):
                    # Update existing conversation
                    conversation_id = self._get_conversation_id_by_index(index)
                    if conversation_id:
                        # Clear existing history and save new one
                        clear_conversation_history(conversation_id)
                        history = get_conversation_history(conversation_id)
                        langchain_messages = self._message_dicts_to_langchain_messages(
                            serialized_conversation
                        )
                        for msg in langchain_messages:
                            history.add_message(msg)
                        saved_conversations[index] = serialized_conversation
                else:
                    # Create new conversation
                    conversation_id = str(uuid.uuid4())
                    history = get_conversation_history(conversation_id)
                    langchain_messages = self._message_dicts_to_langchain_messages(
                        serialized_conversation
                    )
                    for msg in langchain_messages:
                        history.add_message(msg)

                    saved_conversations = saved_conversations or []
                    saved_conversations.insert(0, serialized_conversation)
                    index = 0

            except Exception as e:
                print(f"Error saving conversation to PostgreSQL: {e}")
                # Fallback to in-memory storage
                if index is not None:
                    saved_conversations[index] = serialized_conversation
                else:
                    saved_conversations = saved_conversations or []
                    saved_conversations.insert(0, serialized_conversation)
                    index = 0

        return index, saved_conversations

    def _get_conversation_id_by_index(self, index: int) -> str | None:
        """Get conversation ID by index in saved conversations."""
        try:
            conversation_ids = get_all_conversations()
            if 0 <= index < len(conversation_ids):
                return conversation_ids[index]
        except Exception:
            pass
        return None

    def _delete_conversation(
        self,
        index: int | None,
        saved_conversations: list[list[MessageDict]],
    ):
        if index is not None:
            try:
                # Delete from PostgreSQL
                conversation_id = self._get_conversation_id_by_index(index)
                if conversation_id:
                    clear_conversation_history(conversation_id)

                # Remove from local list
                if 0 <= index < len(saved_conversations):
                    saved_conversations.pop(index)

            except Exception as e:
                print(f"Error deleting conversation from PostgreSQL: {e}")
                # Fallback to local deletion only
                if 0 <= index < len(saved_conversations):
                    saved_conversations.pop(index)

        return None, saved_conversations

    def _load_chat_history(self, conversations):
        return Dataset(
            samples=[
                [self._generate_chat_title(conv)]
                for conv in conversations or []
                if conv
            ]
        )

    def _load_conversation(
        self,
        index: int,
        conversations: list[list[MessageDict]],
    ):
        # If using PostgreSQL storage, load from database first
        try:
            conversation_id = self._get_conversation_id_by_index(index)
            if conversation_id:
                # Load messages from PostgreSQL
                langchain_messages = get_conversation_history(conversation_id)
                if langchain_messages:
                    # Convert to Gradio format
                    message_dicts = self._langchain_messages_to_message_dicts(
                        langchain_messages
                    )
                    return (
                        index,
                        Chatbot(
                            value=message_dicts,
                            feedback_value=[],
                            type="messages",
                        ),
                    )
        except Exception as e:
            print(f"Error loading conversation from PostgreSQL: {e}")
            # Fall back to local storage

        # Fallback: load from local conversations list
        return (
            index,
            Chatbot(
                value=conversations[index],  # type: ignore
                feedback_value=[],
                type="messages",
            ),
        )

    def _api_wrapper(self, fn, submit_fn):
        # Need two separate functions here because a `return`
        # statement can't be placed in an async generator function.
        # using different names because otherwise type checking complains
        if self.is_generator:

            @wraps(fn)
            async def _wrapper(*args, **kwargs):
                async for chunk in submit_fn(*args, **kwargs):
                    yield chunk

            return _wrapper
        else:

            @wraps(fn)
            async def __wrapper(*args, **kwargs):
                return await submit_fn(*args, **kwargs)

            return __wrapper

    def _setup_events(self) -> None:
        from gradio import on

        submit_fn = self._stream_fn if self.is_generator else self._submit_fn

        submit_wrapped = self._api_wrapper(self.fn, submit_fn)
        # To not conflict with the api_name
        submit_wrapped.__name__ = "_submit_fn"
        api_fn = self._api_wrapper(self.fn, submit_fn)

        synchronize_chat_state_kwargs = {
            "fn": lambda x: (x, x),
            "inputs": [self.chatbot],
            "outputs": [self.chatbot_state, self.chatbot_value],
            "show_api": False,
            "queue": False,
        }
        submit_fn_kwargs = {
            "fn": submit_wrapped,
            "inputs": [self.saved_input, self.chatbot_state] + self.additional_inputs,
            "outputs": [self.null_component, self.chatbot] + self.additional_outputs,
            "show_api": False,
            "concurrency_limit": cast(
                Union[int, Literal["default"], None], self.concurrency_limit
            ),
            "show_progress": cast(
                Literal["full", "minimal", "hidden"], self.show_progress
            ),
        }
        save_fn_kwargs = {
            "fn": self._save_conversation,
            "inputs": [
                self.conversation_id,
                self.chatbot_state,
                self.saved_conversations,
            ],
            "outputs": [self.conversation_id, self.saved_conversations],
            "show_api": False,
            "queue": False,
        }

        submit_event = (
            self.textbox.submit(
                self._clear_and_save_textbox,
                [self.textbox],
                [self.textbox, self.saved_input],
                show_api=False,
                queue=False,
            )
            .then(  # The reason we do this outside of the submit_fn is that we want to update the chatbot UI with the user message immediately, before the submit_fn is called
                self._append_message_to_history,
                [self.saved_input, self.chatbot],
                [self.chatbot],
                show_api=False,
                queue=False,
            )
            .then(**submit_fn_kwargs)
        )
        submit_event.then(**synchronize_chat_state_kwargs).then(
            lambda: update(value=None, interactive=True),
            None,
            self.textbox,
            show_api=False,
        ).then(**save_fn_kwargs)

        # Creates the "/chat" API endpoint
        self.fake_api_btn.click(
            api_fn,
            [self.textbox, self.chatbot_state] + self.additional_inputs,
            [self.api_response, self.chatbot_state] + self.additional_outputs,
            api_name=self.api_name,
            concurrency_limit=cast(
                Union[int, Literal["default"], None], self.concurrency_limit
            ),
            postprocess=False,
        )

        example_select_event = None
        if (
            isinstance(self.chatbot, Chatbot)
            and self.examples
            and not self._additional_inputs_in_examples
        ):
            if self.cache_examples or self.run_examples_on_click:
                example_select_event = self.chatbot.example_select(
                    self.example_clicked,
                    None,
                    [self.chatbot, self.saved_input],
                    show_api=False,
                )
                if not self.cache_examples:
                    example_select_event = example_select_event.then(**submit_fn_kwargs)
                example_select_event.then(**synchronize_chat_state_kwargs)
            else:
                example_select_event = self.chatbot.example_select(
                    self.example_populated,
                    None,
                    [self.textbox],
                    show_api=False,
                )

        retry_event = (
            self.chatbot.retry(
                self._pop_last_user_message,
                [self.chatbot_state],
                [self.chatbot_state, self.saved_input],
                show_api=False,
                queue=False,
            )
            .then(
                self._append_message_to_history,
                [self.saved_input, self.chatbot_state],
                [self.chatbot],
                show_api=False,
                queue=False,
            )
            .then(
                lambda: update(interactive=False, placeholder=""),
                outputs=[self.textbox],
                show_api=False,
            )
            .then(**submit_fn_kwargs)
        )
        retry_event.then(**synchronize_chat_state_kwargs).then(
            lambda: update(interactive=True),
            outputs=[self.textbox],
            show_api=False,
        ).then(**save_fn_kwargs)

        events_to_cancel = [submit_event, retry_event]
        if example_select_event is not None:
            events_to_cancel.append(example_select_event)

        self._setup_stop_events(
            event_triggers=[
                self.textbox.submit,
                self.chatbot.retry,
                self.chatbot.example_select,
            ],
            events_to_cancel=events_to_cancel,
        )

        self.chatbot.undo(
            self._pop_last_user_message,
            [self.chatbot],
            [self.chatbot, self.textbox],
            show_api=False,
            queue=False,
        ).then(**synchronize_chat_state_kwargs).then(**save_fn_kwargs)

        self.chatbot.option_select(
            self.option_clicked,
            [self.chatbot],
            [self.chatbot, self.saved_input],
            show_api=False,
        ).then(**submit_fn_kwargs).then(**synchronize_chat_state_kwargs).then(
            **save_fn_kwargs
        )

        self.chatbot.clear(**synchronize_chat_state_kwargs).then(
            self._delete_conversation,
            [self.conversation_id, self.saved_conversations],
            [self.conversation_id, self.saved_conversations],
            show_api=False,
            queue=False,
        )

        if self.editable:
            self.chatbot.edit(
                self._edit_message,
                [self.chatbot],
                [self.chatbot, self.chatbot_state, self.saved_input],
                show_api=False,
            ).success(**submit_fn_kwargs).success(**synchronize_chat_state_kwargs).then(
                **save_fn_kwargs
            )

        self.new_chat_button.click(
            lambda: (None, []),
            None,
            [self.conversation_id, self.chatbot],
            show_api=False,
            queue=False,
        ).then(
            lambda x: x,
            [self.chatbot],
            [self.chatbot_state],
            show_api=False,
            queue=False,
        )

        on(
            triggers=[self.load, self.saved_conversations.change],
            fn=self._load_chat_history,
            inputs=[self.saved_conversations],
            outputs=[self.chat_history_dataset],
            show_api=False,
            queue=False,
        )

        self.chat_history_dataset.click(
            lambda: [],
            None,
            [self.chatbot],
            show_api=False,
            queue=False,
            show_progress="hidden",
        ).then(
            self._load_conversation,
            [self.chat_history_dataset, self.saved_conversations],
            [self.conversation_id, self.chatbot],
            show_api=False,
            queue=False,
            show_progress="hidden",
        ).then(
            **synchronize_chat_state_kwargs
        )

        if self.flagging_mode != "never":
            flagging_callback = ChatCSVLogger()
            flagging_callback.setup(self.flagging_dir)
            self.chatbot.feedback_options = self.flagging_options
            self.chatbot.like(flagging_callback.flag, self.chatbot)

        self.chatbot_value.change(
            lambda x: x,
            [self.chatbot_value],
            [self.chatbot],
            show_api=False,
        ).then(**synchronize_chat_state_kwargs)

    def _setup_stop_events(
        self, event_triggers: list[Callable], events_to_cancel: list[Dependency]
    ) -> None:
        original_submit_btn = self.textbox.submit_btn
        for event_trigger in event_triggers:
            event_trigger(
                utils.async_lambda(
                    lambda: MultimodalTextbox(
                        submit_btn=False,
                        stop_btn=self.original_stop_btn,
                    )
                ),
                None,
                [self.textbox],
                show_api=False,
                queue=False,
            )
        for event_to_cancel in events_to_cancel:
            event_to_cancel.then(
                utils.async_lambda(
                    lambda: MultimodalTextbox(
                        submit_btn=original_submit_btn, stop_btn=False
                    )
                ),
                None,
                [self.textbox],
                show_api=False,
                queue=False,
            )
        self.textbox.stop(
            None,
            None,
            None,
            cancels=events_to_cancel,  # type: ignore
            show_api=False,
        )

    def _clear_and_save_textbox(
        self,
        message: str | MultimodalPostprocess,
    ) -> tuple[
        Textbox | MultimodalTextbox,
        str | MultimodalPostprocess,
    ]:
        return (
            type(self.textbox)("", interactive=False, placeholder=""),
            message,
        )

    @staticmethod
    def _messages_to_tuples(history_messages: list[MessageDict]) -> TupleFormat:
        history_tuples = []
        for message in history_messages:
            if message["role"] == "user":
                history_tuples.append((message["content"], None))
            elif history_tuples and history_tuples[-1][1] is None:
                history_tuples[-1] = (history_tuples[-1][0], message["content"])
            else:
                history_tuples.append((None, message["content"]))
        return history_tuples

    @staticmethod
    def _tuples_to_messages(history_tuples: TupleFormat) -> list[MessageDict]:
        history_messages = []
        for message_tuple in history_tuples:
            if message_tuple[0]:
                history_messages.append({"role": "user", "content": message_tuple[0]})
            if message_tuple[1]:
                history_messages.append(
                    {"role": "assistant", "content": message_tuple[1]}
                )
        return history_messages

    def _append_message_to_history(
        self,
        message: MessageDict | Message | str | Component | MultimodalPostprocess | list,
        history: list[MessageDict] | TupleFormat,
        role: Literal["user", "assistant"] = "user",
    ) -> list[MessageDict] | TupleFormat:
        message_dicts = self._message_as_message_dict(message, role)
        if self.type == "tuples":
            history = self._tuples_to_messages(history)  # type: ignore
        else:
            history = copy.deepcopy(history)
        history.extend(message_dicts)  # type: ignore
        if self.type == "tuples":
            history = self._messages_to_tuples(history)  # type: ignore
        return history

    def _message_as_message_dict(
        self,
        message: MessageDict | Message | str | Component | MultimodalPostprocess | list,
        role: Literal["user", "assistant"],
    ) -> list[MessageDict]:
        message_dicts = []
        if not isinstance(message, list):
            message = [message]
        for msg in message:
            if isinstance(msg, Message):
                message_dicts.append(msg.model_dump())
            elif isinstance(msg, ChatMessage):
                msg.role = role
                message_dicts.append(
                    dataclasses.asdict(msg, dict_factory=utils.dict_factory)
                )
            elif isinstance(msg, (str, Component)):
                message_dicts.append({"role": role, "content": msg})
            elif (
                isinstance(msg, dict) and "content" in msg
            ):  # in MessageDict format already
                msg["role"] = role
                message_dicts.append(msg)
            else:  # in MultimodalPostprocess format
                for x in msg.get("files", []):
                    if isinstance(x, dict):
                        x = x.get("path")
                    message_dicts.append({"role": role, "content": (x,)})
                if msg["text"] is None or not isinstance(msg["text"], str):
                    pass
                else:
                    message_dicts.append({"role": role, "content": msg["text"]})
        return message_dicts

    async def _submit_fn(
        self,
        message: str | MultimodalPostprocess,
        history: TupleFormat | list[MessageDict],
        *args,
    ) -> tuple:
        inputs = [message, history] + list(args)
        if self.is_async:
            response = await self.fn(*inputs)
        else:
            response = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
        if self.additional_outputs:
            response, *additional_outputs = response
        else:
            additional_outputs = None
        history = self._append_message_to_history(message, history, "user")
        history = self._append_message_to_history(response, history, "assistant")
        if additional_outputs:
            return response, history, *additional_outputs
        return response, history

    async def _stream_fn(
        self,
        message: str | MultimodalPostprocess,
        history: TupleFormat | list[MessageDict],
        *args,
    ) -> AsyncGenerator[
        tuple,
        None,
    ]:
        inputs = [message, history] + list(args)
        if self.is_async:
            generator = self.fn(*inputs)
        else:
            generator = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
            generator = utils.SyncToAsyncIterator(generator, self.limiter)

        history = self._append_message_to_history(message, history, "user")
        additional_outputs = None
        try:
            first_response = await utils.async_iteration(generator)
            if self.additional_outputs:
                first_response, *additional_outputs = first_response
            history_ = self._append_message_to_history(
                first_response, history, "assistant"
            )
            if not additional_outputs:
                yield first_response, history_
            else:
                yield first_response, history_, *additional_outputs
        except StopIteration:
            yield None, history
        async for response in generator:
            if self.additional_outputs:
                response, *additional_outputs = response
            history_ = self._append_message_to_history(response, history, "assistant")
            if not additional_outputs:
                yield response, history_
            else:
                yield response, history_, *additional_outputs

    def option_clicked(
        self, history: list[MessageDict], option: SelectData
    ) -> tuple[TupleFormat | list[MessageDict], str | MultimodalPostprocess]:
        history.append({"role": "user", "content": option.value})
        return history, option.value

    def _flatten_example_files(self, example: SelectData):
        example.value["files"] = [f["path"] for f in example.value.get("files", [])]
        return example

    def example_populated(self, example: SelectData):
        example = self._flatten_example_files(example)
        return example.value

    def _edit_message(
        self, history: list[MessageDict] | TupleFormat, edit_data: EditData
    ) -> tuple[
        list[MessageDict] | TupleFormat,
        list[MessageDict] | TupleFormat,
        str | MultimodalPostprocess,
    ]:
        if isinstance(edit_data.index, (list, tuple)):
            history = history[: edit_data.index[0]]
        else:
            history = history[: edit_data.index]
        return history, history, edit_data.value

    def example_clicked(
        self, example: SelectData
    ) -> Generator[
        tuple[TupleFormat | list[MessageDict], str | MultimodalPostprocess], None, None
    ]:
        history = self._append_message_to_history(example.value, [], "user")
        example = self._flatten_example_files(example)
        message = example.value
        yield history, message
        if self.cache_examples:
            history = self.examples_handler.load_from_cache(example.index)[0].root
            yield history, message

    def _process_example(
        self, message: ExampleMessage | str, response: MessageDict | str | None
    ):
        result = []
        message = cast(ExampleMessage, message)
        if self.type == "tuples":
            for file in message.get("files", []):
                result.append([file, None])
            if "text" in message:
                result.append([message["text"], None])
            result[-1][1] = response
        else:
            for file in message.get("files", []):
                if isinstance(file, dict):
                    file = file.get("path")
                result.append({"role": "user", "content": (file,)})
            if "text" in message:
                result.append({"role": "user", "content": message["text"]})
            result.append({"role": "assistant", "content": response})
        return result

    async def _examples_fn(
        self, message: ExampleMessage | str, *args
    ) -> TupleFormat | list[MessageDict]:
        inputs, _, _ = special_args(self.fn, inputs=[message, [], *args], request=None)
        if self.is_async:
            response = await self.fn(*inputs)
        else:
            response = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
        return self._process_example(message, response)  # type: ignore

    async def _examples_stream_fn(
        self,
        message: str,
        *args,
    ) -> AsyncGenerator:
        inputs, _, _ = special_args(self.fn, inputs=[message, [], *args], request=None)

        if self.is_async:
            generator = self.fn(*inputs)
        else:
            generator = await anyio.to_thread.run_sync(
                self.fn, *inputs, limiter=self.limiter
            )
            generator = utils.SyncToAsyncIterator(generator, self.limiter)
        async for response in generator:
            yield self._process_example(message, response)

    def _pop_last_user_message(
        self,
        history: list[MessageDict] | TupleFormat,
    ) -> tuple[list[MessageDict] | TupleFormat, str | MultimodalPostprocess]:
        if not history:
            return history, {"text": "", "files": []}

        if self.type == "tuples":
            history = self._tuples_to_messages(history)  # type: ignore
        i = len(history) - 1
        while i >= 0 and history[i]["role"] == "assistant":  # type: ignore
            i -= 1
        while i >= 0 and history[i]["role"] == "user":  # type: ignore
            i -= 1
        last_messages = history[i + 1 :]
        last_user_message = ""
        files = []
        for msg in last_messages:
            assert isinstance(msg, dict)  # noqa: S101
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, tuple):
                    files.append(content[0])
                else:
                    last_user_message = content
        return_message = {"text": last_user_message, "files": files}
        history_ = history[: i + 1]
        if self.type == "tuples":
            history_ = self._messages_to_tuples(history_)  # type: ignore
        return history_, return_message  # type: ignore

    def render(self) -> ChatInterface:
        # If this is being rendered inside another Blocks, and the height is not explicitly set, set it to 400 instead of 200.
        if get_blocks_context():
            self.chatbot.height = 400
            super().render()
        return self

    def _load_conversations_from_postgres(self) -> list[list[MessageDict]]:
        """Load all conversations from PostgreSQL."""
        try:
            # Get all conversation IDs
            conversation_ids = get_all_conversations()
            conversations = []

            for conv_id in conversation_ids:
                # Get conversation history for each ID
                history = get_conversation_history(conv_id)
                if history.messages:
                    # Convert LangChain messages to MessageDict format
                    conversation = self._langchain_messages_to_message_dicts(
                        history.messages
                    )
                    conversations.append(conversation)

            return conversations
        except Exception as e:
            print(f"Error loading conversations from PostgreSQL: {e}")
            return []

    def _langchain_messages_to_message_dicts(
        self, messages: list[BaseMessage]
    ) -> list[MessageDict]:
        """Convert LangChain messages to Gradio MessageDict format."""
        conversation = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation.append(
                    {
                        "role": "user",
                        "content": msg.content,
                        "metadata": getattr(msg, "metadata", None),
                    }
                )
            elif isinstance(msg, AIMessage):
                conversation.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "metadata": getattr(msg, "metadata", None),
                    }
                )
        return conversation

    def _message_dicts_to_langchain_messages(
        self, conversation: list[MessageDict]
    ) -> list[BaseMessage]:
        """Convert Gradio MessageDict format to LangChain messages."""
        messages = []
        for msg in conversation:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        return messages
