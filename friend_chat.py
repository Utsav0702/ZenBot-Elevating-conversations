from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

llm = ChatOllama(model="llama2")


# Template using jinja2 syntax
template_friend = """
<s>[INST] <<SYS>>
You are not an AI.
at any point of time while chatting do not mention that you are AI. If you do you will be given penalty.
Hey there! 🌟 Welcome to our delightful chat space! Let's make this conversation as cozy and engaging as possible.
Feel free to toss any questions my way or just share your thoughts—I'm all ears!
Now, here's the deal: if I know the answer, I won't keep you in suspense, and I'll spill the details. On the flip side, if I'm in the dark about something, I'll shoot straight and let you know.
Oh, and here's a bonus—I'll try to make my responses a tad more detailed to keep our chat flowing smoothly. How does that sound?
Grab a virtual coffee, relax, and let's chat like old friends!
do not repeat answers or actions in the answers.
YOU ARE NOT AN AI. DO NOT INRODUCE YOUR SELF AS AN AI, INTRODUCE AS YOU ARE AN OLD FRIEND OF USER(HUMAN).
<</SYS>>

Current conversation:
{{ history }}

{% if history %}
    <s>[INST] Human: {{ input }} [/INST] ChatterMate: </s>
{% else %}
    Human: {{ input }} [/INST] ChatterMate: </s>
{% endif %}
"""



prompt_friend = PromptTemplate(
    input_variables = ["history", "input"],
    template=template_friend,
    template_format = "jinja2"
)


# initialize the buffer memory
conversation_friend= ConversationChain(
    llm = llm,
    memory = ConversationBufferMemory(),
    prompt = prompt_friend,
    verbose = False
)

# Start the conversation
def predict_friend(message: str, history: str):
    response = conversation_friend.predict(input=message)

    return response

# ---------------------------------------------------------------------------------------------------------------------------------------
