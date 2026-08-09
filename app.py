import json
import os
import re
import subprocess
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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

def safe_path(rel_path: str) -> str:
    """Ensures paths remain strictly inside WORKSHOP_DIR."""
    rel_path = rel_path.lstrip("/\\")
    full_path = os.path.abspath(os.path.join(WORKSHOP_DIR, rel_path))
    if not full_path.startswith(WORKSHOP_DIR):
        raise HTTPException(status_code=403, detail="Access outside workshop sandbox is forbidden.")
    return full_path

def seed_demo_files():
    main_py = os.path.join(WORKSHOP_DIR, "main.py")
    if not os.path.exists(main_py):
        with open(main_py, "w", encoding="utf-8") as f:
            f.write('# AI Code Editor Sandbox\n\ndef hello():\n    print("Hello from main.py!")\n\nif __name__ == "__main__":\n    hello()\n')
    readme_md = os.path.join(WORKSHOP_DIR, "README.md")
    if not os.path.exists(readme_md):
        with open(readme_md, "w", encoding="utf-8") as f:
            f.write('# Workshop Project\nClick any file in the sidebar to view and edit code in real-time.\nUse the prompt bar below to run AI terminal commands.\n')

seed_demo_files()

app = FastAPI(title="AI Code Editor & Terminal Sandbox")

# Enable CORS for all origins to prevent browser fetch blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class PromptRequest(BaseModel):
    prompt: str

class ExecuteRequest(BaseModel):
    command: str
    explanation: Optional[str] = ""

class SaveFileRequest(BaseModel):
    path: str
    content: str

class CreateFileRequest(BaseModel):
    path: str

class DeleteFileRequest(BaseModel):
    path: str

# -----------------------------------------------------------------------------
# 2. File Operations API (Real Code Editing & File Management)
# -----------------------------------------------------------------------------
@app.get("/api/files")
def list_files():
    """Lists all files and directories in ./workshop."""
    items = []
    for root, dirs, files in os.walk(WORKSHOP_DIR):
        rel_root = os.path.relpath(root, WORKSHOP_DIR)
        for d in sorted(dirs):
            rel_dir = os.path.join(rel_root, d) if rel_root != "." else d
            items.append({"name": d, "path": rel_dir.replace("\\", "/"), "is_dir": True})
        for f in sorted(files):
            rel_file = os.path.join(rel_root, f) if rel_root != "." else f
            items.append({"name": f, "path": rel_file.replace("\\", "/"), "is_dir": False})
    return {"files": items}

@app.get("/api/file/content")
def get_file_content(path: str = Query(...)):
    """Reads raw content of a file inside ./workshop."""
    target = safe_path(path)
    if not os.path.exists(target) or os.path.isdir(target):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()
        return {"path": path, "content": content}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/api/file/save")
def save_file(req: SaveFileRequest):
    """Saves updated file content back to disk inside ./workshop."""
    target = safe_path(req.path)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(req.content)
        return {"status": "success", "message": f"Saved {req.path}"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/api/file/create")
def create_file(req: CreateFileRequest):
    """Creates a new empty file inside ./workshop."""
    target = safe_path(req.path)
    if os.path.exists(target):
        raise HTTPException(status_code=400, detail="File already exists")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write("")
        return {"status": "success", "message": f"Created {req.path}"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/api/file/delete")
def delete_file(req: DeleteFileRequest):
    """Deletes a file from ./workshop."""
    target = safe_path(req.path)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        if os.path.isdir(target):
            import shutil
            shutil.rmtree(target)
        else:
            os.remove(target)
        return {"status": "success", "message": f"Deleted {req.path}"}
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

# -----------------------------------------------------------------------------
# 3. AI Command Translation & Execution Engine
# -----------------------------------------------------------------------------
def local_fallback_command(prompt: str) -> dict:
    p = prompt.lower().strip()
    if "run" in p and ("python" in p or ".py" in p or "main" in p):
        match = re.search(r'([a-zA-Z0-9_\-]+\.py)', prompt)
        fname = match.group(1) if match else "main.py"
        return {"command": f"python3 {fname}", "explanation": f"Executes Python script '{fname}' in terminal."}
    elif "create" in p or "make" in p or "touch" in p:
        if "folder" in p or "directory" in p or "mkdir" in p:
            match = re.search(r'(?:folder|directory|named|called)\s+([a-zA-Z0-9_\-]+)', p)
            dirname = match.group(1) if match else "data"
            return {"command": f"mkdir -p {dirname}", "explanation": f"Creates directory '{dirname}' inside workshop."}
        else:
            match = re.search(r'([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)', prompt)
            filename = match.group(1) if match else "app.py"
            if filename.endswith(".py"):
                cmd = f"echo 'print(\"Hello from {filename}!\")' > {filename}"
            else:
                cmd = f"echo 'Sample content' > {filename}"
            return {"command": cmd, "explanation": f"Creates file '{filename}' inside workshop."}
    elif "list" in p or "ls" in p or "show files" in p:
        return {"command": "ls -la", "explanation": "Lists all files in workshop with permissions and sizes."}
    elif "find" in p or "python" in p:
        return {"command": "find . -name '*.py'", "explanation": "Finds all Python files inside workshop."}
    elif "disk" in p or "space" in p or "usage" in p:
        return {"command": "du -sh *", "explanation": "Displays disk space used by workshop files."}
    elif "read" in p or "cat" in p or "open" in p:
        match = re.search(r'([a-zA-Z0-9_\-]+\.[a-zA-Z0-9]+)', prompt)
        filename = match.group(1) if match else "main.py"
        return {"command": f"cat {filename}", "explanation": f"Prints content of '{filename}' to terminal."}
    else:
        return {"command": f"echo 'Executed prompt: {prompt}'", "explanation": f"Processes request '{prompt}' inside workshop."}

@app.post("/api/ai-command")
def generate_command(req: PromptRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    api_key = os.environ.get("FEATHERLESS_API_KEY")
    if api_key:
        try:
            client = OpenAI(base_url="https://api.featherless.ai/v1", api_key=api_key)
            system_prompt = (
                "You are a command-line AI assistant operating strictly inside a local directory called './workshop'. "
                "Translate the user's request into a single bash command. Respond ONLY with a raw JSON object:\n"
                '{"command": "<shell command>", "explanation": "<1-2 sentence explanation>"}'
            )
            response = client.chat.completions.create(
                model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=10
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                content = "\n".join(lines).strip()
            data = json.loads(content)
            return {"command": str(data.get("command", "")).strip(), "explanation": str(data.get("explanation", "")).strip()}
        except Exception:
            pass

    return local_fallback_command(prompt)

@app.post("/api/execute")
def execute_command(req: ExecuteRequest):
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
        return {"command": cmd, "stdout": "", "stderr": "Execution timed out (30s limit).", "returncode": 124}
    except Exception as err:
        return {"command": cmd, "stdout": "", "stderr": f"Error: {str(err)}", "returncode": 1}

# -----------------------------------------------------------------------------
# 4. Ultra-Robust Web Application Frontend
# -----------------------------------------------------------------------------
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Code Editor & Terminal</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body, html {
            height: 100%;
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
            overflow: hidden;
        }
        .app-container {
            display: flex;
            flex-direction: column;
            height: 100vh;
            width: 100vw;
        }
        .top-nav {
            height: 40px;
            background-color: #252526;
            border-bottom: 1px solid #333333;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 16px;
            font-size: 13px;
        }
        .nav-title { font-weight: bold; color: #ffffff; display: flex; align-items: center; gap: 8px; }
        .nav-badge { background-color: #1e1e1e; border: 1px solid #333333; padding: 2px 8px; border-radius: 3px; font-size: 11px; color: #00ff66; }
        
        .main-layout { display: flex; flex: 1; overflow: hidden; }

        .sidebar {
            width: 260px;
            background-color: #252526;
            border-right: 1px solid #333333;
            display: flex;
            flex-direction: column;
            padding: 12px;
            gap: 12px;
            overflow-y: auto;
        }
        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 11px;
            font-weight: bold;
            color: #888888;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .action-btn-sm {
            background: #2d2d2d;
            color: #cccccc;
            border: 1px solid #3c3c3c;
            padding: 3px 8px;
            font-size: 11px;
            cursor: pointer;
            border-radius: 3px;
            font-family: inherit;
        }
        .action-btn-sm:hover { background: #3e3e42; color: #ffffff; }

        .file-list { display: flex; flex-direction: column; gap: 3px; margin-top: 4px; }
        .file-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 10px;
            font-size: 12px;
            cursor: pointer;
            border-radius: 3px;
            color: #cccccc;
            background-color: #1e1e1e;
            border: 1px solid #333333;
            user-select: none;
        }
        .file-item:hover { background-color: #0e639c; color: #ffffff; border-color: #1177bb; }
        .file-item.active { background-color: #0e639c; color: #ffffff; font-weight: bold; border-color: #007acc; }
        .file-delete-btn { color: #f44747; border: none; background: none; cursor: pointer; font-size: 12px; padding: 0 4px; }
        .file-delete-btn:hover { color: #ffffff; }

        .template-btn {
            background-color: #2d2d2d;
            color: #cccccc;
            border: 1px solid #3c3c3c;
            padding: 8px 10px;
            font-size: 11px;
            font-family: inherit;
            text-align: left;
            cursor: pointer;
            border-radius: 3px;
            transition: all 0.15s ease;
        }
        .template-btn:hover { background-color: #0e639c; color: #ffffff; border-color: #1177bb; }

        .content-area { flex: 1; display: flex; flex-direction: column; overflow: hidden; background-color: #1e1e1e; }
        
        .editor-pane {
            height: 55%;
            display: flex;
            flex-direction: column;
            border-bottom: 1px solid #333333;
        }
        .editor-toolbar {
            height: 36px;
            background-color: #2d2d2d;
            border-bottom: 1px solid #333333;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 12px;
            font-size: 12px;
        }
        .active-file-label { color: #569cd6; font-weight: bold; }
        .save-btn {
            background-color: #238636;
            color: #ffffff;
            border: none;
            padding: 5px 14px;
            font-size: 11px;
            font-family: inherit;
            font-weight: bold;
            cursor: pointer;
            border-radius: 3px;
        }
        .save-btn:hover { background-color: #2ea043; }

        .code-textarea {
            flex: 1;
            background-color: #1e1e1e;
            color: #d4d4d4;
            border: none;
            padding: 12px;
            font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            resize: none;
            outline: none;
            tab-size: 4;
        }

        .terminal-pane {
            height: 45%;
            display: flex;
            flex-direction: column;
            padding: 8px 12px 12px 12px;
            gap: 8px;
            background-color: #1e1e1e;
        }
        .terminal-log {
            flex: 1;
            background-color: #0d0d0d;
            border: 1px solid #333333;
            border-radius: 3px;
            padding: 10px;
            font-size: 12px;
            line-height: 1.5;
            color: #d4d4d4;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .log-prompt { color: #569cd6; font-weight: bold; }
        .log-system { color: #888888; }
        .log-cmd { color: #00ff66; font-weight: bold; }
        .log-stdout { color: #d4d4d4; }
        .log-stderr { color: #f44747; }
        .log-exp { color: #ce9178; font-style: italic; }

        .input-bar { display: flex; gap: 8px; }
        .prompt-input {
            flex: 1;
            background-color: #252526;
            border: 1px solid #333333;
            color: #d4d4d4;
            padding: 10px 12px;
            font-family: inherit;
            font-size: 12px;
            border-radius: 3px;
            outline: none;
        }
        .prompt-input:focus { border-color: #007acc; }
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
        }
        .submit-btn:hover { background-color: #1177bb; }

        .modal-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            background-color: rgba(0, 0, 0, 0.75);
            align-items: center; justify-content: center;
            z-index: 1000;
        }
        .modal-box {
            background-color: #252526;
            border: 1px solid #333333;
            width: 90%; max-width: 550px;
            padding: 20px; border-radius: 4px;
            display: flex; flex-direction: column; gap: 12px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.8);
        }
        .modal-title { font-size: 14px; font-weight: bold; color: #ffffff; }
        .modal-sub { font-size: 12px; color: #aaaaaa; }
        .cmd-box { background-color: #0d0d0d; border: 1px solid #333333; padding: 10px; font-size: 12px; color: #00ff66; border-radius: 3px; white-space: pre-wrap; }
        .exp-box { background-color: #1e1e1e; border: 1px solid #333333; padding: 10px; font-size: 12px; color: #ce9178; border-radius: 3px; }
        .modal-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 4px; }
        .btn-cancel { background-color: #3a3d41; color: #ffffff; border: none; padding: 8px 14px; font-family: inherit; font-size: 12px; cursor: pointer; border-radius: 3px; }
        .btn-approve { background-color: #238636; color: #ffffff; border: none; padding: 8px 16px; font-family: inherit; font-size: 12px; font-weight: bold; cursor: pointer; border-radius: 3px; }
        .btn-approve:hover { background-color: #2ea043; }
    </style>
</head>
<body>

<div class="app-container">
    <div class="top-nav">
        <div class="nav-title">⚡ AI CODE EDITOR & TERMINAL</div>
        <div class="nav-badge">LOCAL SANDBOX: ./workshop</div>
    </div>

    <div class="main-layout">
        <div class="sidebar">
            <div class="section-header">
                <span>Workshop Files</span>
                <div>
                    <button class="action-btn-sm" onclick="createNewFile()">+ New</button>
                    <button class="action-btn-sm" onclick="loadFileList()">🔄</button>
                </div>
            </div>
            <div class="file-list" id="fileList">Loading files...</div>

            <div class="section-header" style="margin-top: 12px;">
                <span>AI Command Templates</span>
            </div>
            <button class="template-btn" onclick="runTemplate('Run main.py in terminal')">▶️ Run main.py in terminal</button>
            <button class="template-btn" onclick="runTemplate('Create a main.py file with print statement')">📝 Create main.py</button>
            <button class="template-btn" onclick="runTemplate('List all files in workshop')">📂 List files (ls -la)</button>
            <button class="template-btn" onclick="runTemplate('Find python files in workshop')">🔍 Find python files</button>
            <button class="template-btn" onclick="runTemplate('Create a folder named data')">📁 Create data folder</button>
        </div>

        <div class="content-area">
            <div class="editor-pane">
                <div class="editor-toolbar">
                    <div>📄 Active File: <span class="active-file-label" id="activeFileName">Select a file from sidebar</span></div>
                    <button class="save-btn" id="saveBtn" onclick="saveActiveFile()" disabled>💾 Save File</button>
                </div>
                <textarea class="code-textarea" id="codeEditor" placeholder="Click any file on the left sidebar to view & edit code..." disabled></textarea>
            </div>

            <div class="terminal-pane">
                <div class="section-header">
                    <span>Terminal Log Output</span>
                    <button class="action-btn-sm" onclick="clearLog()">🗑️ Clear Log</button>
                </div>
                <div class="terminal-log" id="terminalLog">
<span class="log-system">[SYSTEM] Interactive Code Editor & AI Agent initialized.</span>
<span class="log-system">[SYSTEM] Click any file on the left to edit code. Use prompt bar below to execute terminal commands.</span>
--------------------------------------------------------------------------------------------------
</div>
                <div class="input-bar">
                    <input type="text" class="prompt-input" id="promptInput" placeholder="Type prompt (e.g. 'Run main.py' or 'Create test.py')...">
                    <button class="submit-btn" id="submitBtn" onclick="submitPrompt()">GENERATE COMMAND</button>
                </div>
            </div>
        </div>
    </div>
</div>

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
    var activeFilePath = null;
    var pendingCommand = "";
    var pendingExplanation = "";

    function escapeHtml(text) {
        if (!text) return "";
        return String(text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }

    function appendLog(htmlText) {
        var log = document.getElementById('terminalLog');
        if (log) {
            log.innerHTML += htmlText + '\n';
            log.scrollTop = log.scrollHeight;
        }
    }

    function clearLog() {
        var log = document.getElementById('terminalLog');
        if (log) {
            log.innerHTML = '<span class="log-system">[SYSTEM] Terminal log cleared.</span>\n';
        }
    }

    async function loadFileList() {
        var fileListEl = document.getElementById('fileList');
        if (!fileListEl) return;
        
        try {
            var res = await fetch('/api/files');
            var data = await res.json();
            fileListEl.innerHTML = '';
            
            if (!data.files || data.files.length === 0) {
                fileListEl.innerHTML = '<div style="font-size:11px; color:#888888; padding:6px;">(empty directory)</div>';
                return;
            }

            data.files.forEach(function(item) {
                var div = document.createElement('div');
                div.className = 'file-item' + (activeFilePath === item.path ? ' active' : '');
                
                var icon = item.is_dir ? '📁' : '📄';
                div.innerHTML = '<span>' + icon + ' ' + escapeHtml(item.name) + '</span>' +
                                '<button class="file-delete-btn" title="Delete file">🗑️</button>';
                
                div.onclick = function(e) {
                    if (e.target.classList.contains('file-delete-btn')) {
                        e.stopPropagation();
                        deleteFileItem(item.path);
                    } else {
                        openFile(item.path);
                    }
                };

                fileListEl.appendChild(div);
            });
        } catch (err) {
            fileListEl.innerHTML = '<div style="color:#f44747; font-size:11px; padding:6px;">Error loading files</div>';
        }
    }

    async function openFile(filePath) {
        activeFilePath = filePath;
        var nameLabel = document.getElementById('activeFileName');
        var editor = document.getElementById('codeEditor');
        var saveBtn = document.getElementById('saveBtn');

        if (nameLabel) nameLabel.textContent = filePath;

        try {
            var res = await fetch('/api/file/content?path=' + encodeURIComponent(filePath));
            if (!res.ok) throw new Error('Could not read file content');
            var data = await res.json();
            
            if (editor) {
                editor.value = data.content;
                editor.disabled = false;
            }
            if (saveBtn) saveBtn.disabled = false;
        } catch (err) {
            appendLog('<span class="log-stderr">[ERROR] Could not open ' + escapeHtml(filePath) + ': ' + escapeHtml(err.message) + '</span>');
        }
    }

    async function saveActiveFile() {
        if (!activeFilePath) return;
        var editor = document.getElementById('codeEditor');
        if (!editor) return;
        var content = editor.value;

        try {
            var res = await fetch('/api/file/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: activeFilePath, content: content })
            });
            var data = await res.json();
            if (res.ok) {
                appendLog('<span class="log-system">[FILE SAVED] ' + escapeHtml(activeFilePath) + ' successfully updated.</span>');
            } else {
                alert('Save error: ' + (data.detail || 'Unknown error'));
            }
        } catch (err) {
            alert('Save error: ' + err.message);
        }
    }

    async function createNewFile() {
        var path = prompt('Enter new filename (e.g. app.py, test.txt):');
        if (!path) return;
        path = path.trim();
        if (!path) return;
        
        try {
            var res = await fetch('/api/file/create', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: path })
            });
            if (res.ok) {
                await loadFileList();
                await openFile(path);
                appendLog('<span class="log-system">[FILE CREATED] ' + escapeHtml(path) + ' created.</span>');
            } else {
                var data = await res.json();
                alert('Create error: ' + (data.detail || 'Failed to create file'));
            }
        } catch (err) {
            alert('Create error: ' + err.message);
        }
    }

    async function deleteFileItem(filePath) {
        if (!confirm('Are you sure you want to delete ' + filePath + '?')) return;
        try {
            var res = await fetch('/api/file/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: filePath })
            });
            if (res.ok) {
                if (activeFilePath === filePath) {
                    activeFilePath = null;
                    document.getElementById('activeFileName').textContent = 'Select a file from sidebar';
                    var editor = document.getElementById('codeEditor');
                    if (editor) { editor.value = ''; editor.disabled = true; }
                    var saveBtn = document.getElementById('saveBtn');
                    if (saveBtn) saveBtn.disabled = true;
                }
                await loadFileList();
                appendLog('<span class="log-system">[FILE DELETED] ' + escapeHtml(filePath) + ' deleted.</span>');
            }
        } catch (err) {
            alert('Delete error: ' + err.message);
        }
    }

    function runTemplate(promptText) {
        var input = document.getElementById('promptInput');
        if (input) {
            input.value = promptText;
            submitPrompt();
        }
    }

    async function submitPrompt() {
        var input = document.getElementById('promptInput');
        if (!input) return;
        var promptText = input.value.trim();
        if (!promptText) return;

        var btn = document.getElementById('submitBtn');
        if (btn) {
            btn.disabled = true;
            btn.textContent = 'GENERATING...';
        }

        appendLog('\n<span class="log-prompt">&gt; PROMPT:</span> ' + escapeHtml(promptText));
        appendLog('<span class="log-system">[AI] Translating prompt to bash command...</span>');

        try {
            var res = await fetch('/api/ai-command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt: promptText })
            });
            var data = await res.json();
            
            pendingCommand = data.command || "ls -la";
            pendingExplanation = data.explanation || "Executes inside workshop sandbox.";

            var modalCmd = document.getElementById('modalCmd');
            var modalExp = document.getElementById('modalExp');
            var modal = document.getElementById('confirmModal');

            if (modalCmd) modalCmd.textContent = pendingCommand;
            if (modalExp) modalExp.textContent = 'Explanation: ' + pendingExplanation;
            if (modal) modal.style.display = 'flex';
        } catch (err) {
            appendLog('<span class="log-stderr">[ERROR] AI translation failed: ' + escapeHtml(String(err)) + '</span>');
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = 'GENERATE COMMAND';
            }
        }
    }

    async function closeModal(approved) {
        var modal = document.getElementById('confirmModal');
        if (modal) modal.style.display = 'none';
        
        if (!approved) {
            appendLog('<span class="log-system">[CANCELLED] User cancelled command execution.</span>');
            return;
        }

        appendLog('\n<span class="log-cmd">[CONFIRMED] Running command:</span> ' + escapeHtml(pendingCommand));
        appendLog('<span class="log-exp">[EXPLANATION] ' + escapeHtml(pendingExplanation) + '</span>');

        try {
            var res = await fetch('/api/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: pendingCommand, explanation: pendingExplanation })
            });
            var result = await res.json();

            if (result.stdout) {
                appendLog('<span class="log-stdout">[STDOUT]\n' + escapeHtml(result.stdout.trim()) + '</span>');
            }
            if (result.stderr) {
                appendLog('<span class="log-stderr">[STDERR]\n' + escapeHtml(result.stderr.trim()) + '</span>');
            }
            appendLog('<span class="log-system">[EXIT CODE] ' + result.returncode + '</span>');
        } catch (err) {
            appendLog('<span class="log-stderr">[ERROR] Execution failed: ' + escapeHtml(String(err)) + '</span>');
        }

        var input = document.getElementById('promptInput');
        if (input) input.value = '';
        await loadFileList();

        if (activeFilePath) {
            openFile(activeFilePath);
        }
    }

    // Attach Keydown listener safely
    window.addEventListener('DOMContentLoaded', function() {
        var input = document.getElementById('promptInput');
        if (input) {
            input.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    submitPrompt();
                }
            });
        }
        // Initial setup
        loadFileList().then(function() {
            openFile('main.py');
        });
    });
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_CONTENT

if __name__ == "__main__":
    print(f"Starting Web Code Editor Agent on port {PORT}...")
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
