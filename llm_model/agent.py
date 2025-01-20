from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from .model import LlamaModel

class Chat():
    def __init__(self, llm: LlamaModel = None, conversation_chain: ConversationChain = None):
        self.llm = llm
        self.memory = None
        self.conversation_chain = conversation_chain

    def get_message(self, user_input: str):
        # Define the conversation template
        if self.conversation_chain is None:
                prompt_template = """The following is a friendly conversation between a human and an AI.
                The AI is talkative and provides lots of specific details from its context.
                If the AI does not know the answer to a question, it truthfully says it does not know.

                Current conversation:
                {history}
                Human: {input}
                AI Assistant:"""
                prompt = PromptTemplate(input_variables=["history", "input"], template=prompt_template)

                memory = ConversationBufferMemory(ai_prefix="AI Assistant")
                self.conversation_chain = ConversationChain(
                    llm=self.llm,
                    prompt=prompt,
                    memory=memory,
                    verbose=True
                )

        ai_response = self.conversation_chain.run(user_input)
        return ai_response

    def get_conversation_chain(self):
        return self.conversation_chain