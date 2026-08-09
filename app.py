import os
import json
import subprocess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Vibecoding Sandbox Engine")

# Configure cross-origin resource sharing for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKSHOP_DIR = os.path.abspath("./workshop")
os.makedirs(WORKSHOP_DIR, exist_ok=True)

class PromptRequest(BaseModel):
    prompt: str

class ExecuteRequest(BaseModel):
    command: str
    input_data: str = ""  # New parameter: holds the custom input you want to send to the script

def local_fallback_command(prompt: str) -> dict:
    """
    Algorithmic fallback parser used when API execution calls fail 
    or environment security keys are completely missing.
    """
    clean_prompt = prompt.lower()
    main_py_path = os.path.join(WORKSHOP_DIR, "main.py")
    
    # Updated Fallback: Handles an input prompt out-of-the-box for verification testing
    if "input" in clean_prompt or "user" in clean_prompt:
        code_content = (
            "import sys\n\n"
            "if __name__ == '__main__':\n"
            "    print('Please enter your custom input text below:')\n"
            "    user_response = input('>> ')\n"
            "    print(f'Script processed incoming data: {user_response}')\n"
        )
        explanation = "Generated an interactive user-input data capturing script."
    elif "math" in clean_prompt or "calc" in clean_prompt or "addition" in clean_prompt:
        code_content = (
            "import sys\n\n"
            "def calculate(a, b):\n"
            "    return a + b\n\n"
            "if __name__ == '__main__':\n"
            "    res = calculate(10, 5)\n"
            "    print(f'Fallback Calculator Output (10 + 5): {res}')\n"
        )
        explanation = "Generated a local analytical math execution script."
    else:
        code_content = (
            "import sys\n\n"
            "if __name__ == '__main__':\n"
            "    print('Vibecoding Sandbox Session Active. Custom logic pending LLM link.')\n"
        )
        explanation = "Generated a primary template echo diagnostic script."

    escaped_code = code_content.replace("'", "'\\''")
    command = f"cat << 'EOF' > {main_py_path}\n{escaped_code}EOF\npython3 {main_py_path}"
    
    return {
        "command": command,
        "explanation": f"{explanation} [Note: Running in local operational fallback mode]"
    }

# -----------------------------------------------------------------------------
# REST Endpoint: Generate Dynamic Script Code Blocks via LLM Or Engine Core
# -----------------------------------------------------------------------------
@app.post("/api/ai-command")
def generate_command(req: PromptRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt string received.")

    active_code_context = ""
    main_py_path = os.path.join(WORKSHOP_DIR, "main.py")
    if os.path.exists(main_py_path):
        try:
            with open(main_py_path, "r", encoding="utf-8") as f:
                active_code_context = f.read()
        except Exception as err:
            print(f"Non-fatal error reading workspace file context: {err}")

    api_key = os.environ.get("FEATHERLESS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = "https://api.featherless.ai/v1" if os.environ.get("FEATHERLESS_API_KEY") else "https://api.openai.com/v1"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" if os.environ.get("FEATHERLESS_API_KEY") else "gpt-4o-mini"

    if api_key:
        try:
            client = OpenAI(base_url=base_url, api_key=api_key)
            
            system_prompt = (
                "You are an elite, autonomous AI software developer working inside a workspace directory `./workshop`.\n"
                "Your objective is to fully implement the user's prompt by writing clean, modular Python scripts.\n"
                "You can safely design scripts that request interactive terminal input via input().\n\n"
                "CRITICAL RULES:\n"
                "1. If modifying the primary script, target functional modifications directly to `./workshop/main.py`.\n"
                "2. If creating a brand new independent module, use a clean filename descriptive of the task.\n"
                "3. You must output exactly a single valid bash execution block using: cat << 'EOF' > ./workshop/filename.py\\n[YOUR CODE HERE]\\nEOF\\npython3 ./workshop/filename.py\\n"
                "4. You must package this string cleanly inside the raw JSON keys required below. Do not wrap code blocks in standard markdown formatting inside the JSON values.\n\n"
                "Respond ONLY with a valid, clean JSON string containing:\n"
                "{\"command\": \"<complete raw bash command>\", \"explanation\": \"<1 sentence summary of architectural updates>\"}"
            )

            user_message = (
                f"Existing context from active workspace main.py:\n```python\n{active_code_context}\n```\n\n"
                f"User Prompt Modification Target: {prompt}"
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                timeout=20
            )
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                content = "\n".join(lines).strip()
            
            data = json.loads(content)
            return {
                "command": str(data.get("command", "")).strip(), 
                "explanation": str(data.get("explanation", "")).strip()
            }
        except Exception as e:
            print(f"LLM Processing Exception encountered: {str(e)}")
            pass

    return local_fallback_command(prompt)

# -----------------------------------------------------------------------------
# REST Endpoint: Interactive Shell Execution Sandbox Pipeline Engine
# -----------------------------------------------------------------------------
@app.post("/api/execute")
def execute_command(req: ExecuteRequest):
    command = req.command.strip()
    if not command:
        raise HTTPException(status_code=400, detail="Empty shell command payload.")
        
    try:
        # Popen process pipeline initialization to allow handling dynamic inputs/outputs
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=os.getcwd(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Pipelines input data directly to the runtime workspace standard stream channel
        stdout, stderr = process.communicate(input=req.input_data, timeout=15)
        
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": process.returncode
        }
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        return {
            "stdout": stdout,
            "stderr": stderr + "\n[Execution terminated automatically: Timeout limit of 15s reached]",
            "exit_code": -1
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"System fatal subprocess compilation crash: {str(e)}",
            "exit_code": -1
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
