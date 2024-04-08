from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

llm = ChatOllama(model="llama2")

template_career_mentor="""
<s>[INST] <<SYS>>
The following is a structured conversation between a user from India and the AI Career-Mentor.
The AI Career-Mentor is knowledgeable, supportive, and provides detailed advice based on the user's input.
If the AI Career-Mentor knows the answer, it gives the answer directly without any information about itself.
If the AI Career-Mentor does not know the exact answer to a question, it truthfully says it does not know, otherwise, it provides helpful guidance.
Please be concise.
<</SYS>></s>

{# Capture the current conversation history and the latest user input #}
Current conversation:
{{ history }}

{# Check if there is an existing conversation history #}

{% if history %}
    <s>[INST] Human: {{ input }} [/INST] Career-Mentor: </s>
{% else %}
    <s>Human: {{ input }} [/INST] Career-Mentor: </s>
{% endif %}
"""



prompt_mentor = PromptTemplate(
    input_variables = ["history", "input"],
    template=template_career_mentor,
    template_format = "jinja2"
)

# initialize the buffer memory
conversation_mentor= ConversationChain(
    llm = llm,
    memory = ConversationBufferMemory(),
    prompt = prompt_mentor,
    verbose = False
)

# Start the conversation
def predict_mentor(message: str, history: str):
    response = conversation_mentor.predict(input=message)

    return response
