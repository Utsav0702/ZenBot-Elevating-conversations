from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

llm = ChatOllama(model="llama2")



# ---------------------------------------------------------------------------------------------------------------------------------------

# Template using jinja2 syntax
template_normal = """
<s>[INST] <<SYS>>
The following is a friendly conversation between a human and AI.
The AI is talkative and provides lots of specific details from its context.
If the AI knows the answer give Answer directly without any infromation about himself.
If the AI does not know the exact answer to a question, it truthfully says it does not know, otherwise give the answers.
Please be concise.
<</SYS>>
Current conversation:
{{ history }}

{% if history %}
    <s>[INST] Human: {{ input }} [/INST] Lucia: </s>
{% else %}
    Human: {{ input }} [/INST] Lucia: </s>
{% endif %}

"""



prompt_normal = PromptTemplate(
    input_variables = ["history", "input"],
    template=template_normal,
    template_format = "jinja2"
)

# initialize the buffer memory
conversation_normal= ConversationChain(
    llm = llm,
    memory = ConversationBufferMemory(),
    prompt = prompt_normal,
    verbose = False
)

# Start the conversation
def predict_normal(message: str, history: str):
    response = conversation_normal.predict(input=message)

    return response
# ---------------------------------------------------------------------------------------------------------------------------------------
