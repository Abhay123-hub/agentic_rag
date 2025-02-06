```markdown
# Agentic RAG Chatbot with LangGraph

This project implements an agentic Retrieval Augmented Generation (RAG) chatbot using LangGraph, LangChain, and a Large Language Model (LLM) from Groq.  It retrieves relevant information from a specified set of web pages to answer user queries.  The chatbot uses a grading mechanism to determine if the retrieved documents are relevant and employs a question rewriter to improve query understanding.

## Features

* **Retrieval Augmented Generation (RAG):** Combines the power of LLMs with retrieved context from external sources.
* **Agentic Approach:** Integrates an agent to manage the information retrieval and response generation process.
* **Document Grading:**  Evaluates the relevance of retrieved documents to the user's question.
* **Question Rewriting:**  Refines the user's question to improve retrieval accuracy.
* **Streamlit Integration:** Provides an interactive web interface for users to interact with the chatbot.

## Technologies Used

* **LangGraph:** Orchestrates the complex workflow of the agentic RAG pipeline.
* **LangChain:** Provides components for LLM interaction, prompt engineering, and document loading.
* **Groq LLM:** Powers the language understanding and generation capabilities of the chatbot.
* **Hugging Face Embeddings:** Creates vector representations of text for semantic search.
* **Chroma:** Stores and manages the vector database of document embeddings.
* **Streamlit:** Builds the interactive web application.
* **Python:** The primary programming language.
* **WebBaseLoader:** Loads documents from the web.
* **RecursiveCharacterTextSplitter:** Splits the loaded documents into chunks.
* **dotenv:** Loads environment variables from a `.env` file.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Abhay123-hub/agentic_rag.git(https://www.google.com/search?q=https://github.com/Abhay123-hub/agentic_rag.git)  #
   cd agentic_rag.git
   ```

2. Create a virtual environment (recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   Create a `requirements.txt` file with the following content:

   ```
   langchain
   langgraph
   langchain-community
   langchain-huggingface
   langchain-groq
   transformers
   streamlit
   chromadb
   tiktoken
   python-dotenv
   ```

4. Create a `.env` file in the project directory and add your Groq API key:

   ```
   GROQ_API_KEY=YOUR_GROQ_API_KEY
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run streamlit_app.py  # Replace streamlit_app.py with your Streamlit file name
   ```

2. Open your web browser and navigate to the URL displayed by Streamlit (usually `http://localhost:8501`).

3. Enter your question in the text box and click "Get Answer."

## Project Structure

```
agentic_rag/
├── chatbot.py         # Contains the chatbot logic and LangGraph workflow.
├── streamlit_app.py  # Contains the Streamlit web application code.
├── requirements.txt   # Lists the project dependencies.
├── .env               # Stores environment variables (including API keys).
└── ...                # Other files (e.g., logos, data files).
```

## Code Explanation (Key Parts)

* **`chatbot.py`:** This file defines the core chatbot logic. The `chatbot` class contains the methods for different nodes in the LangGraph workflow:
    * `AI_Assistant`: Handles initial user input and responses after query rewriting.
    * `grade_documents`: Determines the relevance of retrieved documents.
    * `generate`: Generates the final response using the RAG prompt.
    * `question_rewriter`: Rewrites the user's question.
    * `__call__`: Defines the LangGraph workflow and compiles it.

* **`streamlit_app.py`:** This file creates the Streamlit web interface.  It takes user input, passes it to the `chatbot` instance, and displays the response.  It also handles the styling and layout of the application.

## Further Development

* **Improved Prompt Engineering:** Experiment with different prompts to optimize the chatbot's performance.
* **More Data Sources:** Integrate additional data sources beyond the initial web pages.
* **Error Handling:** Implement more robust error handling to gracefully handle unexpected situations.
* **User Authentication:** Add user authentication to personalize the chatbot experience.
* **Deployment:** Deploy the chatbot to a cloud platform for wider accessibility.

## Contributing

Contributions are welcome!  Please open an issue or submit a pull request.


