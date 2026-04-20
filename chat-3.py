from typing import List
from ctransformers import AutoModelForCausalLM

llm2 = AutoModelForCausalLM.from_pretrained("TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGUF", model_file="wizardlm-1.0-uncensored-llama2-13b.Q2_K.gguf")

def get_prompt(instruction: str, history: List[str] | None = None) -> str:

    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    if history is not None:
        prompt += f"This is the convo history: {''.join(history)}. \nNow answer the question: "
    print(f"Prompt: {prompt}")
    return prompt

history = []

question = "What is the capital of India?"
answer = ""

for word in llm2(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()
history.append(answer)

question = "And which is the capital of the United States?"

for word in llm2(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()

print(history)
