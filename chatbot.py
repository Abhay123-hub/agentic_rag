import warnings
warnings.filterwarnings("ignore")

from typing import Annotated,Literal,Sequence,TypedDict
from langchain import hub
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel,Field
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition,ToolNode
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import START,StateGraph,END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import uuid
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
from langgraph.checkpoint.memory import MemorySaver


from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = ChatGroq(model_name = "Gemma2-9b-It")
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
]
docs = [WebBaseLoader(url).load() for url in urls]
documents = [d for url in docs for d in url]

text_splitter = RecursiveCharacterTextSplitter(chunk_size =200,chunk_overlap=50)
docs_split = text_splitter.split_documents(documents)

vectorstore = Chroma.from_documents(
    documents=docs_split,
    embedding=embeddings,
    collection_name="rag-chrome"
)

retriever = vectorstore.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "retriever",
    "search and return information about ai agents,llm ,prompt engineering and agentic ai related queries"
)

tools = [retriever_tool]
retrieve = ToolNode(tools) 

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class Grade(BaseModel):
    binary_score:str = Field(description="gives relevence score 'yes' or 'no' ")

class chatbot:
    def __init__(self):
        pass
    def AI_Assistant(self,state:AgentState):
        messages = state["messages"]
        if len(messages)>1: ## means the question is coming to ai assistant again or after the query rewriter, so now i need to simply generate the response of this question
            question = messages[-1].content
            prompt = PromptTemplate(
                template = """ You are a helpful assistant whatever question is given to you , you will give answer of that question very 
                carefully.here is the question:{question}
                                """,
            input_variables=["question"]
            )

            chain = prompt | llm
            response = chain.invoke(question)
            return {"messages":[response]}
        else:
            llm_with_tool = llm.bind_tools(tools)
            messages = state["messages"]
            question = messages[-1].content
            response = llm_with_tool.invoke(question)
            return {"messages":[response]}
    def grade_documents(self,state:AgentState):
        messages = state["messages"]
        question = messages[0].content
        documents = messages[-1].content
        
        prompt = PromptTemplate(
            template = """ You are a helpful assistant.Given the question and document to you.
            You need to find out that the provided question is relevant to documents or not.You need to see it carefully.
            if the question is relevant to the documents return 'yes' and if the question is not relevant to the documents
            then return 'no'
            The question is:{question} and the documents:{documents} 

                        """,
            input_variables=["question","documents"]
            

        )

        llm_structured = llm.with_structured_output(Grade)
        chain = prompt | llm_structured
        binary_score = chain.invoke({"question":question,"documents":documents})
        if binary_score.binary_score == 'yes':
            return 'generate'
        else:
            return 'question_rewriter'
    
    def generate(self,state:AgentState):
        messages = state["messages"]
        question = messages[0].content
        documents = messages[-1].content

        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = prompt | llm
        response = rag_chain.invoke({"question":question,"context":documents})
        return {"messages":[response]}
    def question_rewriter(self,state:AgentState):
        messages = state["messages"]
        question = messages[0].content

        prompt = PromptTemplate(
            template = """ You are a helpful assistant. You will be given a question.Your job is to know the intention of the question.
            then update the question . do not add extra thing while upodating the question, if you are not able to update the question leave it as it is.
            the question is:{question}
                            """,
            input_variables=["question"]

        )

        chain = prompt|llm
        response = chain.invoke({"question":question})
        return {"messages":[response]}
    
    def __call__(self):
        memory = MemorySaver()
        workflow = StateGraph(AgentState)
        ## defining nodes of the graph
        workflow.add_node("AI_ASSISTANT",self.AI_Assistant)
        workflow.add_node("retriever",retrieve)
        workflow.add_node("generate",self.generate)
        workflow.add_node("question_rewriter",self.question_rewriter)
        ## now after defining all the nodes mow i need to join all the nodes
        workflow.add_edge(START,"AI_ASSISTANT")
        workflow.add_conditional_edges("AI_ASSISTANT",tools_condition,{"tools":"retriever",END:END})
        workflow.add_conditional_edges("retriever",self.grade_documents,{"generate":"generate","question_rewriter":"question_rewriter"})
        workflow.add_edge("question_rewriter","AI_ASSISTANT")
        workflow.add_edge("generate",END)

        app = workflow.compile(checkpointer=memory)
        self.graph = app
        return self.graph

if __name__ == "__main__":
    mybot = chatbot()
    workflow = mybot()
    inputs = {"messages":["who is PM of India?"]}
    response = workflow.invoke(inputs,config=config)
    print(response["messages"][-1].content)

