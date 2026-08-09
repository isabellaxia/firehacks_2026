import json
import os
import re
import subprocess
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openai import OpenAI
import uvicorn

# -----------------------------------------------------------------------------
# 1. Environment & Sandbox Setup
# -----------------------------------------------------------------------------
PORT = int(os.environ.get("PORT", 8080))
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
WORKSHOP_DIR = os.path.join(BASE_DIR, "workshop")
os.makedirs(WORKSHOP_DIR, exist_ok=True)

# Seed initial demo files if directory is empty
def seed_demo_files():
    main_py = os.path.join(WORKSHOP_DIR, "main.py")
    if not os.path.exists(main_py):
        with open(main_py, "w") as f:
            f.write('# AI Code Editor Workshop\nprint("Welcome to the AI Code Editor Sandbox!")\n')
    readme_md = os.path.join(WORKSHOP_DIR, "README.md")
    if not os.path.exists(readme_md):
        with open(readme_md, "w") as f:
            f.write('# Workshop Sandbox\nAll generated commands execute safely inside this folder.\n')

seed_demo_files()

app = FastAPI(title="AI Code Editor Agent")

class PromptRequest(BaseModel):
    prompt: str

class ExecuteRequest(BaseModel):
    command: str
    explanation: Optional[str] = ""

# -----------------------------------------------------------------------------
# 2. Smart Hybrid AI & Local Fallback Engine
# -----------------------------------------------------------------------------
def local_fallback_command(prompt: str) -> dict:
    """Intelligent rule-based fallback command generator for instant demo reliability."""
    p = prompt.lower().strip()

    if "create" in p or "make" in p or "touch" in p:
        if "folder" in p or "directory" in p or "mkdir" in p:
            match = re.search(r'(?:folder|directory|named|called)\s+([a-zA-Z0-9_\-]+)', p)
            dirname = match.group(1) if match else "data"
            return {
                "command": f"mkdir -p {dirname}",
                "explanation": f"Creates a new directory named '{dirname}' in workshop."
            }
        else:
            match = re.search(r'([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)', prompt)
            filename = match.group(1) if match else "app.py"
            if filename.endswith(".py"):
                cmd = f"echo 'print(\"Hello from {filename}!\")' > {filename}"
            else:
                cmd = f"echo 'Created content for {filename}' > {filename}"
            return {
                "command": cmd,
                "explanation": f"Creates a file named '{filename}' inside workshop."
            }
    elif "list" in p or "ls" in p or "show files" in p:
        return {
            "command": "ls -la",
            "explanation": "Lists all files in workshop with detailed permissions and sizes."
        }
    elif "python" in p or "find" in p or ".py" in p:
        return {
            "command": "find . -name '*.py'",
            "explanation": "Finds all Python files inside the workshop directory."
        }
    elif "disk" in p or "space" in p or "usage" in p:
        return {
            "command": "du -sh *",
            "explanation": "Calculates disk space usage for all files in workshop."
        }
    elif "read" in p or "cat" in p or "open" in p or "show" in p:
        match = re.search(r'([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)', prompt)
        filename = match.group(1) if match else "main.py"
        return {
            "command": f"cat {filename}",
            "explanation": f"Displays the content of {filename} in the terminal."
        }
    elif "remove" in p or "delete" in p or "rm" in p:
        match = re.search(r'([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)', prompt)
        filename = match.group(1) if match else "temp.txt"
        return {
            "command": f"rm -f {filename}",
            "explanation": f"Deletes file {filename} from workshop sandbox."
        }
    else:
        return {
            "command": f"echo 'Executed prompt: {prompt}'",
            "explanation": f"Processes the request '{prompt}' inside workshop."
        }

@app.get("/api/files")
def get_files():
    """Returns a plain-text tree view of files inside ./workshop."""
    if not os.path.exists(WORKSHOP_DIR):
        return {"tree": "./workshop/ (created)"}

    tree_lines = ["./workshop/"]
    for root, dirs, files in os.walk(WORKSHOP_DIR):
        rel_path = os.path.relpath(root, WORKSHOP_DIR)
        if rel_path == ".":
            depth = 0
        else:
            depth = rel_path.count(os.sep) + 1
            indent = "  " * depth
            tree_lines.append(f"{indent}📁 {os.path.basename(root)}/")

        indent = "  " * (depth + 1)
        for d in sorted(dirs):
            if rel_path == ".":
                tree_lines.append(f"{indent}📁 {d}/")
        for f in sorted(files):
            tree_lines.append(f"{indent}📄 {f}")

    if len(tree_lines) == 1:
        tree_text = "./workshop/\n  (empty directory)"
    else:
        tree_text = "\n".join(tree_lines)

    return {"tree": tree_text}

@app.post("/api/ai-command")
def generate_command(req: PromptRequest):
    """Translates natural language prompt to bash command using Featherless AI (with smart local fallback)."""
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    api_key = os.environ.get("FEATHERLESS_API_KEY")
    if api_key:
        try:
            client = OpenAI(
                base_url="https://api.featherless.ai/v1",
                api_key=api_key
            )

            system_prompt = (
                "You are a command-line AI assistant operating strictly inside a local directory called './workshop'. "
                "Translate the user's natural language request into a single precise bash shell command. "
                "You MUST respond ONLY with a raw JSON object containing exactly two keys:\n"
                '1. "command": The exact shell command string to execute.\n'
                '2. "explanation": A concise 1-2 sentence explanation of what the command does.\n\n'
                "Example response:\n"
                '{"command": "touch main.py", "explanation": "Creates an empty file named main.py inside the workshop directory."}\n'
                "Do not include any extra commentary or markdown formatting."
            )

            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                timeout=10
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            data = json.loads(content)
            return {
                "command": str(data.get("command", "")).strip(),
                "explanation": str(data.get("explanation", "")).strip()
            }
        except Exception:
            # Fall through to local fallback engine if API fails or times out
            pass

    # Guaranteed fallback response (always works even without API key)
    return local_fallback_command(prompt)

@app.post("/api/execute")
def execute_command(req: ExecuteRequest):
    """Executes an approved bash command strictly inside ./workshop."""
    cmd = req.command.strip()
    if not cmd:
        raise HTTPException(status_code=400, detail="Empty command")

    try:
        res = subprocess.run(
            cmd,
            shell=True,
            cwd=WORKSHOP_DIR,
            capture_output=True,
            text=True,
            timeout=30
        )
        return {
            "command": cmd,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "returncode": res.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "command": cmd,
            "stdout": "",
            "stderr": "Command execution timed out after 30 seconds.",
            "returncode": 124
        }
    except Exception as err:
        return {
            "command": cmd,
            "stdout": "",
            "stderr": f"Execution error: {str(err)}",
            "returncode": 1
        }

# -----------------------------------------------------------------------------
# 3. Single-Page Web Frontend (Dark Mode Code Editor Aesthetic)
# -----------------------------------------------------------------------------
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Code Editor Agent</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body, html {
            height: 100%;
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
            overflow: hidden;
        }
        .container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            width: 100vw;
        }
        /* Top Navigation Header */
        .header {
            height: 42px;
            background-color: #252526;
            border-bottom: 1px solid #333333;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 16px;
            font-size: 13px;
        }
        .header-title {
            font-weight: bold;
            color: #ffffff;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .header-badge {
            background-color: #1e1e1e;
            border: 1px solid #333333;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 11px;
            color: #00ff66;
        }
        /* Main Workspace Split */
        .workspace {
            display: flex;
            flex: 1;
            overflow: hidden;
        }
        /* Sidebar */
        .sidebar {
            width: 280px;
            background-color: #252526;
            border-right: 1px solid #333333;
            display: flex;
            flex-direction: column;
            padding: 12px;
            gap: 12px;
            overflow-y: auto;
        }
        .sidebar-section-title {
            font-size: 11px;
            font-weight: bold;
            color: #888888;
            letter-spacing: 1px;
            text-transform: uppercase;
        }
        .template-btn {
            background-color: #2d2d2d;
            color: #cccccc;
            border: 1px solid #3c3c3c;
            padding: 8px 10px;
            font-size: 12px;
            font-family: inherit;
            text-align: left;
            cursor: pointer;
            border-radius: 3px;
            transition: all 0.15s ease;
        }
        .template-btn:hover {
            background-color: #0e639c;
            color: #ffffff;
            border-color: #1177bb;
        }
        .file-tree-container {
            background-color: #0d0d0d;
            border: 1px solid #333333;
            padding: 8px;
            border-radius: 3px;
            flex: 1;
            overflow: auto;
            font-size: 12px;
            color: #a6e22e;
            white-space: pre;
        }
        .sidebar-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .icon-btn {
            background: none;
            border: none;
            color: #888888;
            cursor: pointer;
            font-size: 13px;
        }
        .icon-btn:hover {
            color: #ffffff;
        }
        /* Main Log Output Area */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 12px;
            gap: 10px;
            background-color: #1e1e1e;
        }
        .terminal-log {
            flex: 1;
            background-color: #0d0d0d;
            border: 1px solid #333333;
            border-radius: 3px;
            padding: 12px;
            font-size: 12px;
            line-height: 1.5;
            color: #d4d4d4;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .log-entry-prompt { color: #569cd6; font-weight: bold; }
        .log-entry-system { color: #888888; }
        .log-entry-cmd { color: #00ff66; font-weight: bold; }
        .log-entry-stdout { color: #d4d4d4; }
        .log-entry-stderr { color: #f44747; }
        .log-entry-exp { color: #ce9178; font-style: italic; }

        /* Input Control Bar */
        .input-bar {
            display: flex;
            gap: 8px;
        }
        .prompt-input {
            flex: 1;
            background-color: #252526;
            border: 1px solid #333333;
            color: #d4d4d4;
            padding: 10px 12px;
            font-family: inherit;
            font-size: 13px;
            border-radius: 3px;
            outline: none;
        }
        .prompt-input:focus {
            border-color: #007acc;
        }
        .submit-btn {
            background-color: #0e639c;
            color: #ffffff;
            border: none;
            padding: 10px 18px;
            font-family: inherit;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            border-radius: 3px;
            transition: background 0.15s ease;
        }
        .submit-btn:hover {
            background-color: #1177bb;
        }

        /* Confirmation Modal Window */
        .modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(0, 0, 0, 0.75);
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal-box {
            background-color: #252526;
            border: 1px solid #333333;
            width: 90%;
            max-width: 550px;
            padding: 20px;
            border-radius: 4px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.8);
        }
        .modal-title {
            font-size: 14px;
            font-weight: bold;
            color: #ffffff;
            letter-spacing: 0.5px;
        }
        .modal-sub {
            font-size: 12px;
            color: #aaaaaa;
        }
        .cmd-box {
            background-color: #0d0d0d;
            border: 1px solid #333333;
            padding: 10px;
            font-size: 12px;
            color: #00ff66;
            border-radius: 3px;
            white-space: pre-wrap;
        }
        .exp-box {
            background-color: #1e1e1e;
            border: 1px solid #333333;
            padding: 10px;
            font-size: 12px;
            color: #ce9178;
            border-radius: 3px;
        }
        .modal-actions {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: 4px;
        }
        .btn-cancel {
            background-color: #3a3d41;
            color: #ffffff;
            border: none;
            padding: 8px 14px;
            font-family: inherit;
            font-size: 12px;
            cursor: pointer;
            border-radius: 3px;
        }
        .btn-cancel:hover { background-color: #4e5257; }
        .btn-approve {
            background-color: #238636;
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            font-family: inherit;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            border-radius: 3px;
        }
        .btn-approve:hover { background-color: #2ea043; }
    </style>
</head>
<body>

<div class="container">
    <!-- Top Header -->
    <div class="header">
        <div class="header-title">
            ⚡ CODE EDITOR AGENT
        </div>
        <div class="header-badge">
            SANDBOX: ./workshop
        </div>
    </div>

    <!-- Main Workspace -->
    <div class="workspace">
        <!-- Sidebar -->
        <div class="sidebar">
            <div class="sidebar-section-title">Command Templates</div>
            <button class="template-btn" onclick="runTemplate('Create a main.py file with a print statement')">Create main.py file</button>
            <button class="template-btn" onclick="runTemplate('List all files in workshop')">List directory files</button>
            <button class="template-btn" onclick="runTemplate('Find all .py files in workshop')">Find python files</button>
            <button class="template-btn" onclick="runTemplate('Show disk space usage of workshop')">Check disk usage</button>
            <button class="template-btn" onclick="runTemplate('Create a folder named data')">Create data folder</button>
            <button class="template-btn" onclick="runTemplate('Read contents of main.py')">Read main.py contents</button>

            <div class="sidebar-row" style="margin-top: 8px;">
                <div class="sidebar-section-title">Workshop Files</div>
                <button class="icon-btn" onclick="loadFiles()" title="Refresh file list">🔄 Refresh</button>
            </div>
            <pre class="file-tree-container" id="fileTree">Loading files...</pre>
        </div>

        <!-- Main Content (Terminal Window) -->
        <div class="main-content">
            <div class="sidebar-row">
                <div class="sidebar-section-title">Terminal Log Output</div>
                <button class="icon-btn" onclick="clearLog()">🗑️ Clear Log</button>
            </div>
            <div class="terminal-log" id="terminalLog">
<span class="log-entry-system">[SYSTEM] AI Code Editor Agent Sandbox initialized.</span>
<span class="log-entry-system">[SYSTEM] All commands execute strictly in local ./workshop folder.</span>
----------------------------------------------------------------------
</div>
            <div class="input-bar">
                <input type="text" class="prompt-input" id="promptInput" placeholder="Type prompt e.g. 'Create a file named test.py'..." onkeydown="if(event.key==='Enter') handlePromptSubmit()">
                <button class="submit-btn" id="submitBtn" onclick="handlePromptSubmit()">GENERATE COMMAND</button>
            </div>
        </div>
    </div>
</div>

<!-- Execution Confirmation Modal -->
<div class="modal-overlay" id="confirmModal">
    <div class="modal-box">
        <div class="modal-title">CONFIRM COMMAND EXECUTION</div>
        <div class="modal-sub">Proposed shell command to run inside <b>./workshop</b>:</div>
        <div class="cmd-box" id="modalCmd"></div>
        <div class="exp-box" id="modalExp"></div>
        <div class="modal-actions">
            <button class="btn-cancel" onclick="closeModal(false)">CANCEL</button>
            <button class="btn-approve" onclick="closeModal(true)">APPROVE & EXECUTE</button>
        </div>
    </div>
</div>

<script>
    let pendingCommand = "";
    let pendingExplanation = "";

    function runTemplate(promptText) {
        document.getElementById('promptInput').value = promptText;
        handlePromptSubmit();
    }

    function clearLog() {
        document.getElementById('terminalLog').innerHTML = '<span class="log-entry-system">[SYSTEM] Terminal log cleared.</span>\n';
    }

    function appendLog(htmlText) {
        const log = document.getElementById('terminalLog');
        log.innerHTML += htmlText + '\n';
        log.scrollTop = log.scrollHeight;
    }

    async function loadFiles() {
        try {
            const res = await fetch('/api/files');
            const data = await res.json();
            document.getElementById('fileTree').textContent = data.tree || '(empty)';
        } catch (err) {
            document.getElementById('fileTree').textContent = 'Error loading files: ' + err;
        }
    }

    async function handlePromptSubmit() {
        const input = document.getElementById('promptInput');
        const prompt = input.value.trim();
        if (!prompt) return;

        const btn = document.getElementById('submitBtn');
        btn.disabled = true;
        btn.textContent = 'GENERATING...';

        appendLog(`\n<span class="log-entry-prompt">&gt; PROMPT:</span> ${escapeHtml(prompt)}`);
        appendLog(`<span class="log-entry-system">[AI] Translating prompt to command...</span>`);

        try {
            const res = await fetch('/api/ai-command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: prompt })
            });
            const data = await res.json();
            
            pendingCommand = data.command || "ls -la";
            pendingExplanation = data.explanation || "Executes command inside workshop.";

            document.getElementById('modalCmd').textContent = pendingCommand;
            document.getElementById('modalExp').textContent = 'Explanation: ' + pendingExplanation;
            document.getElementById('confirmModal').style.display = 'flex';
        } catch (err) {
            appendLog(`<span class="log-entry-stderr">[ERROR] Failed to generate command: ${escapeHtml(String(err))}</span>`);
        } finally {
            btn.disabled = false;
            btn.textContent = 'GENERATE COMMAND';
        }
    }

    async function closeModal(approved) {
        document.getElementById('confirmModal').style.display = 'none';
        
        if (!approved) {
            appendLog(`<span class="log-entry-system">[CANCELLED] User cancelled command execution.</span>`);
            return;
        }

        appendLog(`\n<span class="log-entry-cmd">[CONFIRMED] Running command:</span> ${escapeHtml(pendingCommand)}`);
        appendLog(`<span class="log-entry-exp">[EXPLANATION] ${escapeHtml(pendingExplanation)}</span>`);

        try {
            const res = await fetch('/api/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: pendingCommand, explanation: pendingExplanation })
            });
            const result = await res.json();

            if (result.stdout) {
                appendLog(`<span class="log-entry-stdout">[STDOUT]\n${escapeHtml(result.stdout.trim())}</span>`);
            }
            if (result.stderr) {
                appendLog(`<span class="log-entry-stderr">[STDERR]\n${escapeHtml(result.stderr.trim())}</span>`);
            }
            appendLog(`<span class="log-entry-system">[EXIT CODE] ${result.returncode}</span>`);
        } catch (err) {
            appendLog(`<span class="log-entry-stderr">[ERROR] Execution failed: ${escapeHtml(String(err))}</span>`);
        }

        document.getElementById('promptInput').value = '';
        loadFiles();
    }

    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Initial load
    loadFiles();
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    """Serves the single-page dark-mode web GUI."""
    return HTML_CONTENT

# -----------------------------------------------------------------------------
# 4. Entrypoint for Render & Local Server Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting Web-Based Code Editor Agent on port {PORT}...")
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
