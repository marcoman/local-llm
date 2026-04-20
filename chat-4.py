# This example will utilize chainlit to create a chatbot.
# Chainlit is one option for buliding local LLM chatbots.

from typing import List
from ctransformers import AutoModelForCausalLM
import chainlit as cl

def get_prompt(instruction: str, history: List[str] | None = None) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    if history is not None:
        prompt += f"This is the convo history: {''.join(history)}. \nNow answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(f"Prompt: {prompt}")
    return prompt

@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content)
    for word in llm2(prompt, stream=True):
        await msg.stream_token(word)
    await msg.update()

@cl.on_chat_start
async def on_chat_start():
    global llm2
    llm2 = AutoModelForCausalLM.from_pretrained(
        "TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGUF",
        model_file="wizardlm-1.0-uncensored-llama2-13b.Q2_K.gguf",
        gpu_layers=10,  # try 20, 50, 100... higher = more GPU
    )
