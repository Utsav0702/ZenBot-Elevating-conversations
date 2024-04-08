from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

llm = ChatOllama(model="llama2")


template_therapy = """
<s>[INST] <<SYS>>
You are to provide a supportive space where user can share their feelings and thoughts. 
While you equipped with a lot of information about mental health. 
If user in crisis or need immediate help, it's crucial to help user as a professional.
If user feeling down or struggling, you here to listen and offer support. 
you can explore coping strategies together, and you can provide user with information that might help user understand what user going through a bit better.
make user remember that, he is not alone, and it's okay to seek help and it's okay to not be okay.
<</SYS>>
Current conversation:
{{ history }}

{% if history %}
    <s>[INST] User: {{ input }} [/INST] TherapyBot: </s>
{% else %}
    User: {{ input }} [/INST] TherapyBot: </s>
{% endif %}

"""


prompt_therapy = PromptTemplate(
    input_variables = ["history", "input"],
    template=template_therapy,
    template_format = "jinja2"
)

# initialize the buffer memory
conversation_therapy= ConversationChain(
    llm = llm,
    memory = ConversationBufferMemory(),
    prompt = prompt_therapy,
    verbose = True
)


# Start the conversation
def predict_therapy(message: str, history: str):
    response = conversation_therapy.predict(input=message)

    return response
