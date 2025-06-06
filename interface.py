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
)
from gradio.components.chatbot import (
    ChatMessage,
    Message,
    MessageDict,
)
from gradio.components.multimodal_textbox import MultimodalPostprocess, MultimodalValue
from gradio.context import get_blocks_context
from gradio.events import Dependency, EditData, SelectData
from gradio.flagging import ChatCSVLogger
from gradio.helpers import special_args, update
from gradio.i18n import I18nData
from gradio.layouts import Column, Group, Row
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
        fn: Generator[str, Any, None],
        *,
        editable: bool = False,
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
        self.concurrency_limit = concurrency_limit
        if isinstance(fn, ChatInterface):
            self.fn = fn.fn
        else:
            self.fn = fn
        # Since fn is always a Generator[str, Any, None], we don't need async/generator checks
        self.is_async = False
        self.is_generator = True
        self.editable = editable
        self.fill_height = fill_height
        self.autoscroll = autoscroll
        self.autofocus = autofocus
        self.title = title
        self.description = description
        self.show_progress = show_progress

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

            self._setup_events()

    def _render_header(self):
        if self.title:
            Markdown(
                f"<h1 style='text-align: center; margin-bottom: 1rem'>{self.title}</h1>"
            )
        if self.description:
            Markdown(self.description)

    def _render_history_area(self):
        with Column(scale=1, min_width=320):
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
        self.chatbot = Chatbot(
            label="Chatbot",
            scale=1,
            height=400 if self.fill_height else None,
            type="messages",
            autoscroll=self.autoscroll,
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
        conversation_id: str | None,
        conversation: list[MessageDict],
        saved_conversations: list[list[MessageDict]],
    ):
        if conversation:
            try:
                serialized_conversation = self.serialize_components(conversation)

                # Save to PostgreSQL
                if conversation_id:
                    # Update existing conversation
                    clear_conversation_history(conversation_id)
                    history = get_conversation_history(conversation_id)
                    langchain_messages = self._message_dicts_to_langchain_messages(
                        serialized_conversation
                    )
                    for msg in langchain_messages:
                        history.add_message(msg)

                    # Update in local list if it exists
                    for i, conv in enumerate(saved_conversations):
                        if self._get_conversation_title(
                            conv
                        ) == self._get_conversation_title(serialized_conversation):
                            saved_conversations[i] = serialized_conversation
                            break
                    else:
                        # If not found in local list, add it
                        saved_conversations.insert(0, serialized_conversation)
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

            except Exception as e:
                print(f"Error saving conversation to PostgreSQL: {e}")
                # Fallback to in-memory storage
                if conversation_id:
                    # Update existing conversation in local list
                    for i, conv in enumerate(saved_conversations):
                        if self._get_conversation_title(
                            conv
                        ) == self._get_conversation_title(serialized_conversation):
                            saved_conversations[i] = serialized_conversation
                            break
                    else:
                        saved_conversations.insert(0, serialized_conversation)
                else:
                    # Create new conversation
                    conversation_id = str(uuid.uuid4())
                    saved_conversations = saved_conversations or []
                    saved_conversations.insert(0, serialized_conversation)

        return conversation_id, saved_conversations

    def _get_conversation_title(self, conversation: list[MessageDict]) -> str:
        """Helper method to get a consistent title for a conversation."""
        return self._generate_chat_title(conversation)

    def _delete_conversation(
        self,
        conversation_id: str | None,
        saved_conversations: list[list[MessageDict]],
    ):
        if conversation_id:
            try:
                # Delete from PostgreSQL
                clear_conversation_history(conversation_id)

                # Remove from local list by finding matching conversation
                title_to_remove = None
                for conv in saved_conversations:
                    # Try to match by checking if this conversation belongs to the ID
                    try:
                        history = get_conversation_history(conversation_id)
                        if (
                            not history.messages
                        ):  # If successfully cleared, remove from local list
                            title_to_remove = self._get_conversation_title(conv)
                            break
                    except Exception:
                        pass

                if title_to_remove:
                    saved_conversations = [
                        conv
                        for conv in saved_conversations
                        if self._get_conversation_title(conv) != title_to_remove
                    ]

            except Exception as e:
                print(f"Error deleting conversation from PostgreSQL: {e}")

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
        # Get conversation ID from the conversations stored in PostgreSQL
        try:
            conversation_ids = get_all_conversations()
            if 0 <= index < len(conversation_ids):
                conversation_id = conversation_ids[index]
                print("Loading conversation ID:", conversation_id)

                # Load messages from PostgreSQL
                history = get_conversation_history(conversation_id)
                if history.messages:
                    # Convert to Gradio format
                    message_dicts = self._langchain_messages_to_message_dicts(
                        history.messages
                    )
                    return (
                        conversation_id,
                        Chatbot(
                            value=message_dicts,
                            feedback_value=[],
                            type="messages",
                        ),
                    )
        except Exception as e:
            print(f"Error loading conversation from PostgreSQL: {e}")

        # Fallback: load from local conversations list
        if 0 <= index < len(conversations):
            # Generate a new conversation ID for local conversations
            conversation_id = str(uuid.uuid4())
            return (
                conversation_id,
                Chatbot(
                    value=conversations[index],
                    feedback_value=[],
                    type="messages",
                ),
            )

        return None, Chatbot(value=[], feedback_value=[], type="messages")

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
            "inputs": [self.saved_input, self.chatbot_state],
            "outputs": [self.null_component, self.chatbot],
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
        ).then(
            **save_fn_kwargs
        )  # Creates the "/chat" API endpoint
        self.fake_api_btn.click(
            api_fn,
            [self.textbox, self.chatbot_state],
            [self.api_response, self.chatbot_state],
            api_name=self.api_name,
            concurrency_limit=cast(
                Union[int, Literal["default"], None], self.concurrency_limit
            ),
            postprocess=False,
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

        self._setup_stop_events(
            event_triggers=[
                self.textbox.submit,
                self.chatbot.retry,
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
        # Trigger initial load of chat history
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

    def _append_message_to_history(
        self,
        message: MessageDict | Message | str | Component | MultimodalPostprocess | list,
        history: list[MessageDict],
        role: Literal["user", "assistant"] = "user",
    ) -> list[MessageDict]:
        message_dicts = self._message_as_message_dict(message, role)
        history = copy.deepcopy(history)
        history.extend(message_dicts)
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
        history: list[MessageDict],
        *args,
    ) -> tuple:
        # Get current conversation_id from state - don't create new one if history exists
        conversation_id = self.conversation_id.value
        if not conversation_id and history:
            # If we have history but no conversation_id, this means we're continuing an existing conversation
            # Try to find the conversation ID from existing conversations
            try:
                conversation_ids = get_all_conversations()
                if conversation_ids:
                    # Use the first conversation ID as we're likely continuing it
                    conversation_id = conversation_ids[0]
                    print(f"Reusing existing conversation ID: {conversation_id}")
            except Exception:
                pass

        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            print(f"Created new conversation ID: {conversation_id}")

        inputs = [message, conversation_id] + list(args)
        # Since fn is always a Generator[str, Any, None], collect all chunks
        generator = await anyio.to_thread.run_sync(
            self.fn, *inputs, limiter=self.limiter
        )
        response_chunks = []
        for chunk in generator:
            response_chunks.append(chunk)
        response = "".join(response_chunks)

        history = self._append_message_to_history(message, history, "user")
        history = self._append_message_to_history(response, history, "assistant")
        return response, history

    async def _stream_fn(
        self,
        message: str | MultimodalPostprocess,
        history: list[MessageDict],
        *args,
    ) -> AsyncGenerator[
        tuple,
        None,
    ]:
        # Get current conversation_id from state - don't create new one if history exists
        conversation_id = self.conversation_id.value
        if not conversation_id and history:
            # If we have history but no conversation_id, this means we're continuing an existing conversation
            # Try to find the conversation ID from existing conversations
            try:
                conversation_ids = get_all_conversations()
                if conversation_ids:
                    # Use the first conversation ID as we're likely continuing it
                    conversation_id = conversation_ids[0]
                    print(f"Reusing existing conversation ID: {conversation_id}")
            except Exception:
                pass

        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            print(f"Created new conversation ID: {conversation_id}")

        print(f"Streaming Conversation ID: {conversation_id}")
        inputs = [message, conversation_id] + list(args)
        # Since fn is always a Generator[str, Any, None], get the generator
        generator = await anyio.to_thread.run_sync(
            self.fn, *inputs, limiter=self.limiter
        )
        # Convert to async iterator
        generator = utils.SyncToAsyncIterator(generator, self.limiter)

        history = self._append_message_to_history(message, history, "user")
        accumulated_response = ""
        try:
            first_response = await utils.async_iteration(generator)
            accumulated_response = first_response
            history_ = self._append_message_to_history(
                accumulated_response, history, "assistant"
            )
            yield first_response, history_
        except StopIteration:
            yield None, history
            return

        async for response in generator:
            accumulated_response += response
            history_ = self._append_message_to_history(
                accumulated_response, history, "assistant"
            )
            yield response, history_

    def option_clicked(
        self, history: list[MessageDict], option: SelectData
    ) -> tuple[list[MessageDict], str | MultimodalPostprocess]:
        history.append({"role": "user", "content": option.value})
        return history, option.value

    def _edit_message(self, history: list[MessageDict], edit_data: EditData) -> tuple[
        list[MessageDict],
        list[MessageDict],
        str | MultimodalPostprocess,
    ]:
        if isinstance(edit_data.index, (list, tuple)):
            history = history[: edit_data.index[0]]
        else:
            history = history[: edit_data.index]
        return history, history, edit_data.value

    def _pop_last_user_message(
        self,
        history: list[MessageDict],
    ) -> tuple[list[MessageDict], str | MultimodalPostprocess]:
        if not history:
            return history, {"text": "", "files": []}

        i = len(history) - 1
        while i >= 0 and history[i]["role"] == "assistant":
            i -= 1
        while i >= 0 and history[i]["role"] == "user":
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
            print(f"IDS: {conversation_ids}")
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
