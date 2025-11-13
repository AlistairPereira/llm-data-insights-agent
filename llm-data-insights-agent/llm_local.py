import subprocess
import textwrap

def run_llm(prompt: str, model: str = "llama3.2") -> str:
    # Clean up indentation / extra spaces in the prompt
    prompt = textwrap.dedent(prompt).strip()

    # Call the local LLM via Ollama
    result = subprocess.run(
        ["ollama", "run", model],          # e.g. `ollama run llama3.2`
        input=prompt.encode("utf-8"),      # send the prompt to the model
        stdout=subprocess.PIPE,            # capture model output
        stderr=subprocess.PIPE,            # capture errors / warnings
    )

    # If Ollama prints any error text, show it in the console
    if result.stderr:
        print("⚠ Ollama stderr:", result.stderr.decode("utf-8", errors="ignore"))

    # Return the model's generated text as a Python string
    return result.stdout.decode("utf-8", errors="ignore")
