
import streamlit as st
from transformers import pipeline
from chatbot import chatbot
import uuid

config = {"configurable": {"thread_id": str(uuid.uuid4())}}
mybot = chatbot()
workflow = mybot()

# Set up the Streamlit UI with improved aesthetics
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")  # Wide layout for better use of space

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        font-family: sans-serif;
        background-color: #f4f4f4; /* Light background color */
    }
    .container {
        max-width: 900px;
        margin: 20px auto;
        padding: 20px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* Subtle shadow */
    }
    .title {
        color: #333; /* Darker title color */
        text-align: center;
        margin-bottom: 20px;
    }
    .input-area {
        margin-bottom: 20px;
    }
    .input-area textarea {
        width: 100%;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        resize: vertical; /* Allow vertical resizing of the textarea */
        min-height: 100px;
    }
    .button-area {
        text-align: center;
    }
    .answer-area {
        margin-top: 20px;
        background-color: #e6f2ff; /* Light blue background for answers */
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ccebff; /* Slightly darker border */
        white-space: pre-wrap; /* Preserve whitespace and wrap text */
    }

    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown("<div class='container'>", unsafe_allow_html=True)  # Start container
st.markdown("<h1 class='title'>ChatBot With LangGraph 📚 😄 🤖 🧐</h1>", unsafe_allow_html=True)

# Input area with larger text area
st.markdown("<div class='input-area'>", unsafe_allow_html=True)  # Start input area
question = st.text_area("Enter your question here 🤖", height=150) # Larger text area
st.markdown("</div>", unsafe_allow_html=True) # end input area


input = {"messages": [question]}

# Button area
st.markdown("<div class='button-area'>", unsafe_allow_html=True)
if st.button("Get Answer"):
    if input:
        with st.spinner("Thinking..."): # Add a spinner while processing
            response = workflow.invoke(input, config=config)
            st.markdown("<div class='answer-area'>", unsafe_allow_html=True) # start answer area
            st.markdown(f"**Answer:** {response['messages'][-1].content}", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True) # end answer area
st.markdown("</div>", unsafe_allow_html=True)  # End button area
st.markdown("</div>", unsafe_allow_html=True)  # End container
