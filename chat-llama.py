from ctransformers import AutoModelForCausalLM

llm2 = AutoModelForCausalLM.from_pretrained("TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGUF", model_file="wizardlm-1.0-uncensored-llama2-13b.Q2_K.gguf")

def get_prompt(instruction: str, history: list[str] | None = None) -> str:

    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n{instruction}\n\n### Response:\n"
    print(f"Prompt created: {prompt}")
    return prompt

question = "What is the capital of India?"

for word in llm2(get_prompt(question), stream=True):
    print(word, end="", flush=True)
print()

