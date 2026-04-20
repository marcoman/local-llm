from langchain_community.llms import CTransformers

llm = CTransformers(
    model="TheBloke/WizardLM-1.0-Uncensored-Llama2-13B-GGUF",
    model_file="wizardlm-1.0-uncensored-llama2-13b.Q2_K.gguf",
    gpu_layers=10,  # try 20, 50, 100... higher = more GPU
    model_type="llama2",
    max_new_tokens=25,
)

# llm = CTransformers(
#     model="zoltanctoth/orca_mini_3B-GGUF",
#     model_file="orca-mini-3b.q4_0.gguf",
#     model_type="llama2",
#     max_new_tokens=25,
# )

print(llm("What is the capital of India?"))