import streamlit as st
from parquool import Agent
from openai.types.responses import ResponseTextDeltaEvent


async def stream(prompt):
    async for event in st.session_state.agent.stream(prompt):
        # We'll print streaming delta if available
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            yield event.data.delta
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                yield f"{event.item.raw_item.name} - {event.item.raw_item.arguments}\n\n"
            elif event.item.type == "tool_call_output_item":
                yield event.item.output
            else:
                pass


st.title("Test Agent")

if not st.session_state.get("agent"):
    st.session_state.agent = Agent(
        tools=[Agent.google_search, Agent.read_url]
    )

st.session_state.messages = st.session_state.agent.get_conversation()

for message in st.session_state.messages:
    if message.get("role") == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message.get("role") == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"][0]["text"])
    elif message.get("type") == "function_call":
        with st.chat_message("assistant"):
            with st.expander(message["name"]):
                st.code(message["arguments"])
    elif message.get("type") == "function_call_output":
        with st.chat_message("assistant"):
            with st.expander("Expand to see the result"):
                st.code(message["output"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = st.write_stream(stream(prompt))
