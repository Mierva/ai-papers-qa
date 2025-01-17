from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from model import LlamaModel


llm = LlamaModel()
llm._initialize_model()

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant"),
)
conversation.predict(input="Hi there!")