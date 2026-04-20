from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

# This prompt gives you a long answer:

#prompt = "Tell me the name of the capital of India."
#print(llm(prompt))

# This prompt tries to get you a shorter answer:
#prompt = "What is the capital of India in 2 words."
#print(llm(prompt))


def first_prompt():
    # This prompt streams the answer word by word:
    prompt = "The name of the capital of India is"

    for word in llm(prompt, stream=True):
        print(word, end="", flush=True)
    print()

first_prompt()
