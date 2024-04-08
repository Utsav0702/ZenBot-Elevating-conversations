from normal_chat import predict_normal
from friend_chat import predict_friend
from mentor_chat import predict_mentor
from therapy_chat import predict_therapy
from girlfriend_chat import predict_girlfriend
from boyfriend_chat import predict_boyfriend

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("Welcome to Zenbot! 🤖")        
    
    with gr.Tab("Normal Chat"):
        Normalchat = gr.ChatInterface(
            fn = predict_normal,
            title = "Normal Chat",)
        
    with gr.Tab("Friend Chat"):
        Friendchat = gr.ChatInterface(
            fn=predict_friend,
            title="Friend Chat",
        )
    
    with gr.Tab("Therapy Chat"):
        Therapychat = gr.ChatInterface(
            predict_therapy,
            title="Therapy Chat",
        )
        
    with gr.Tab("Mentor Chat"):
        Mentorchat = gr.ChatInterface(
            predict_mentor,
            title="Mentor Chat",
        )
        
    with gr.Tab("Girlfriend Chat"):
        Girlfriendchat = gr.ChatInterface(
            predict_girlfriend,
            title="Girlfriend Chat",
        )
    
    with gr.Tab("Boyfriend Chat"):
        Boyfriendchat = gr.ChatInterface(
            predict_boyfriend,
            title="Boyfriend Chat",
        )
        
demo.launch(
    inbrowser = True,
    height = 600,
    inline = False,
    share = True,
    width = 800,
    debug=True,
)