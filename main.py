from interface import ChatInterface
import os
from server.v1 import gen_ai_response, config


def stream_chat(prompt, session_id):
    print(f"SESSION ID: {session_id}!")
    print(f"PROMPT: {prompt}")
    for chunk in gen_ai_response(
        prompt=prompt["text"],
        session_id=session_id,
    ):
        # print(chunk, end="", flush=True)
        yield chunk


demo = ChatInterface(
    fn=stream_chat,
    # type="messages",
)

if __name__ == "__main__":
    demo.launch(
        auth=("rayat.ass", "ppooii12"),
        debug=config.debug,
    )
