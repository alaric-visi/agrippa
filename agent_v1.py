import json
import os
import subprocess
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import re
import requests
import inspect
import ast
from functools import wraps, lru_cache

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PROXY_URL = os.getenv("AI_PROXY_URL", "http://sandbox-proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1200"))
AGENT_MODELS = ["zai-org/GLM-4.5-FP8"]

# Tunable limits
REQUEST_TIMEOUT = int(os.getenv("AGENT_REQUEST_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "2"))
RETRY_BASE = float(os.getenv("AGENT_RETRY_BASE_DELAY", "1.0"))
MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "100"))
MAX_CONSECUTIVE_ERRORS = int(os.getenv("AGENT_MAX_CONSEC_ERRORS", "8"))
USE_CHAT_FALLBACK = os.getenv("AGENT_USE_CHAT_FALLBACK", "0") in {"1", "true", "True"}

# Consolidated prompts
PROMPTS = {
    "format": textwrap.dedent("""
    Use this format:
    next_thought: What you're thinking and planning to do next
    next_tool_name: Tool name to use
    next_tool_args: Tool arguments in JSON format

    Example:
    next_thought: "I need to read the main file to understand the structure"
    next_tool_name: "read_file"
    next_tool_args: {"file_path": "app.py"}
    """),
    
    "system": textwrap.dedent("""
    You are a super senior python software engineer. Your job is to:
    1. Understand the problem statement thoroughly
    2. Explore the codebase to understand current implementation  
    3. Split task into smaller ones, solve each small task by generating and applying small code edits
    4. Make the necessary code changes to solve the problem
    5. Never create or modify test files to verify your changes

    Available tools:
    {tools_docs}

    Work systematically: understand first, then explore, then solve.
    Make minimal changes that correctly address the problem.
    Only modify Python files (.py), never create or modify test files.
    DO NOT repeat the same tool call with the same arguments.

    Critical rules:
    - Never output or propose a patch/diff. Make real edits using the tools.
    - After ANY file modification, immediately call run_syntax_check on that file before proceeding.

    {format_prompt}
    """),
    
    "instance": "Problem to solve:\n{problem_statement}\n\nLet's solve this step by step!",
    
    "stop": textwrap.dedent("""
    DO NOT generate `observation:` in your response. It will be provided by user for you.
    Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
    DO NOT repeat the same tool call with the same arguments.
    """)
}

# =============================================================================
# Utility functions
# =============================================================================

def handle_errors(func):
    """Decorator to handle common exceptions in tools"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (FileNotFoundError, PermissionError, Exception) as e:
            error_type = type(e).__name__
            if error_type == "FileNotFoundError":
                return f"File not found: {e}"
            elif error_type == "PermissionError":
                return f"Permission denied: {e}"
            else:
                return f"Error in {func.__name__}: {e}"
    return wrapper

def run_subprocess(cmd: List[str], timeout: int = 30) -> Tuple[str, str, int]:
    """Run subprocess with timeout"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", 1
    except Exception as e:
        return "", f"Command failed: {e}", 1

# =============================================================================
# Core Tools
# =============================================================================

@handle_errors
def read_file(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read file contents with optional line range support for large files."""
    if not os.path.exists(file_path):
        return f"File {file_path} does not exist"
    
    with open(file_path, "r", encoding="utf-8") as f:
        if start_line or end_line:
            lines = f.readlines()
            start = max(0, (start_line or 1) - 1)
            end = min(len(lines), end_line or len(lines))
            content = ''.join(lines[start:end])
            return f"Lines {start + 1}-{end} of {file_path}:\n{content}"
        content = f.read()
    
    # Efficient truncation for large files
    if len(content) > 50000:
        lines = content.split('\n')
        if len(lines) > 1000:
            return '\n'.join(lines[:500] + ['... (file truncated) ...'] + lines[-500:])
    
    return content

@handle_errors
def edit_file(file_path: str, old_code: str, new_code: str) -> str:
    """Replace specific code in a file with new code."""
    if not os.path.exists(file_path):
        return f"File {file_path} does not exist"
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if old_code not in content:
        return f"Code to replace not found in {file_path}.\nFile content:\n{content[:2000]}..."
    
    count = content.count(old_code)
    if count > 1:
        return f"Found {count} occurrences of the code. Please be more specific."
    
    new_content = content.replace(old_code, new_code)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    
    return f"Successfully edited {file_path}"

@handle_errors
def edit_file_regex(file_path: str, pattern: str, replacement: str, count: int = 1, flags: str = "") -> str:
    """Regex-based edit: replace pattern with replacement. Set count=-1 for all."""
    if not os.path.exists(file_path):
        return f"File {file_path} does not exist"
    try:
        re_flags = 0
        if "I" in flags: re_flags |= re.IGNORECASE
        if "M" in flags: re_flags |= re.MULTILINE
        if "S" in flags: re_flags |= re.DOTALL
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        new_content, n = re.subn(pattern, replacement, content, count=0 if count == -1 else count, flags=re_flags)
        if n == 0:
            return f"No matches for pattern in {file_path}"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"✅ Regex edit applied to {file_path} ({n} replacement(s))"
    except re.error as e:
        return f"Regex error: {e}"

@handle_errors
def create_file(file_path: str, content: str) -> str:
    """Create a new file with specified content."""
    if 'test' in file_path.lower():
        return "Error: Cannot create test files"
    
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Successfully created {file_path}"

@handle_errors
def insert_code_at_location(file_path: str, line_number: int, code: str, position: str = "after") -> str:
    """Insert code at specific line numbers."""
    if not os.path.exists(file_path):
        return f"File {file_path} does not exist"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if not (1 <= line_number <= len(lines)):
        return f"Invalid line number {line_number}. File has {len(lines)} lines."
    
    if not code.endswith('\n'):
        code += '\n'
    
    insert_index = line_number - 1 if position == "before" else line_number
    lines.insert(insert_index, code)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    return f"✅ Successfully inserted code at line {line_number} ({position}) in {file_path}"

@handle_errors
def list_files(directory: str = ".") -> str:
    """List all Python files in a directory and its subdirectories."""
    stdout, stderr, returncode = run_subprocess(["find", directory, "-name", "*.py", "-type", "f"])
    
    if returncode != 0:
        return f"Error listing files: {stderr}"
    
    files = [f.strip() for f in stdout.split('\n') 
             if f.strip() and '__pycache__' not in f and not f.startswith('./test')]
    
    return '\n'.join(sorted(files)) if files else f"No Python files found in {directory}"

@handle_errors
def search_codebase(search_term: str) -> str:
    """Search for a term in the entire codebase with context."""
    cmd = ["bash", "-c", f"grep -rn -B 5 -A 5 --include='*.py' . -e '{search_term}'"]
    stdout, stderr, returncode = run_subprocess(cmd)
    
    if returncode != 0 or not stdout:
        return f"'{search_term}' not found in the codebase."
    
    # Limit output to prevent overwhelming
    if len(stdout) > 10000:
        lines = stdout.split('\n')[:100]
        stdout = '\n'.join(lines) + "\n\n... output truncated ..."
    
    return stdout

@handle_errors
def search_in_file(file_path: str, search_term: str) -> str:
    """Search for a term in the specified file with context."""
    cmd = ["bash", "-c", f"grep -rn -B 5 -A 5 '{search_term}' {file_path}"]
    stdout, stderr, returncode = run_subprocess(cmd)
    
    if returncode != 0 or not stdout:
        return f"'{search_term}' not found in file '{file_path}'"
    
    return stdout

@handle_errors
def run_syntax_check(file_path: str) -> str:
    """Check Python syntax without executing code."""
    if not os.path.exists(file_path) or not file_path.endswith('.py'):
        return f"Invalid Python file: {file_path}"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    try:
        compile(source_code, file_path, 'exec')
        return f"✅ Syntax check passed for {file_path}"
    except SyntaxError as e:
        return f"❌ Syntax Error in {file_path}:\nLine {e.lineno}: {e.text.strip() if e.text else ''}\nError: {e.msg}"
    except Exception as e:
        return f"❌ Compilation Error in {file_path}: {e}"

@handle_errors
def analyze_code_structure(file_path: str) -> str:
    """Parse and analyze Python code structure using AST."""
    if not os.path.exists(file_path) or not file_path.endswith('.py'):
        return f"Invalid Python file: {file_path}"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()
    
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return f"Syntax error in {file_path}: {e}"
    
    imports, classes, functions, globals_vars = [], [], [], []
    
    # Single pass through AST
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend([f"import {alias.name}" for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.extend([f"from {module} import {alias.name}" for alias in node.names])
        elif isinstance(node, ast.ClassDef):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            classes.append({'name': node.name, 'methods': methods, 'line': node.lineno})
        elif isinstance(node, ast.FunctionDef):
            # Check if it's a top-level function (not a method)
            parent_classes = [n for n in ast.walk(tree) 
                            if isinstance(n, ast.ClassDef) and node in getattr(n, 'body', [])]
            if not parent_classes:
                args = [arg.arg for arg in node.args.args]
                functions.append({'name': node.name, 'args': args, 'line': node.lineno})
    
    # Top-level assignments
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    globals_vars.append(target.id)
    
    # Build analysis
    analysis = [f"Code Structure Analysis for {file_path}:\n"]
    
    for section_name, items, formatter in [
        ("📦 Imports", imports[:10], lambda x: f"  {x}"),
        ("🏗️  Classes", classes, lambda x: f"  {x['name']} (line {x['line']})" + 
         (f"\n    Methods: {', '.join(x['methods'][:5])}" if x['methods'] else "")),
        ("🔧 Functions", functions, lambda x: f"  {x['name']}({', '.join(x['args'])}) (line {x['line']})"),
        ("🌐 Global Variables", globals_vars[:10], lambda x: x)
    ]:
        if items:
            analysis.append(f"\n{section_name} ({len(items)}):")
            if section_name.startswith("🌐"):
                analysis.append(f"  {', '.join(items)}")
            else:
                analysis.extend([formatter(item) for item in items[:10]])
                if len(items) > 10:
                    analysis.append(f"  ... and {len(items) - 10} more")
    
    total_lines = len(source_code.split('\n'))
    analysis.append(f"\n📊 Metrics: {total_lines} lines, {len(classes)} classes, {len(functions)} functions")
    
    return '\n'.join(analysis)

@handle_errors
def discover_relevant_files(file_path: str) -> str:
    """Discover files related to a given file using directory structure and imports."""
    if not os.path.exists(file_path):
        return f"File {file_path} does not exist"
    
    base_dir = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).replace('.py', '')
    
    results = [f"Discovering files related to: {file_path}"]
    
    # Same directory files
    if base_dir:
        stdout, _, returncode = run_subprocess(["find", base_dir, "-name", "*.py", "-type", "f"])
        if returncode == 0:
            same_dir = [f for f in stdout.strip().split('\n') 
                       if f and f != file_path and 'test' not in f.lower()]
            if same_dir:
                results.append(f"\nSame Directory ({len(same_dir[:10])} files):")
                results.extend([f"  {f}" for f in same_dir[:10] if os.path.exists(f)])
    
    # Import relationships
    import_patterns = [f"from {base_name} import", f"import {base_name}"]
    import_results = []
    
    for pattern in import_patterns:
        stdout, _, returncode = run_subprocess(
            ["bash", "-c", f"grep -r --include='*.py' '{pattern}' . | grep -v test"]
        )
        if returncode == 0 and stdout.strip():
            files = [line.split(':')[0] for line in stdout.strip().split('\n') if ':' in line]
            import_results.extend(files)
    
    if import_results:
        unique_imports = list(set(import_results))[:10]
        results.append(f"\nImport Relations ({len(unique_imports)} files):")
        results.extend([f"  {f}" for f in unique_imports if os.path.exists(f)])
    
    return '\n'.join(results) if len(results) > 1 else "No related files found"

@handle_errors
def get_changes() -> str:
    """Show all current code changes made so far."""
    git_patch, _ = get_final_git_patch()
    return git_patch if git_patch.strip() else "No changes detected"

def finish() -> str:
    """Signal that the task is complete."""
    return "TASK_COMPLETE"

def get_final_git_patch() -> Tuple[str, List[str]]:
    """Get the git patch of all changes in the working directory."""
    logs = []
    
    stdout, stderr, _ = run_subprocess(["bash", "-c", "shopt -s globstar ; git add **/*.py"])
    logs.append(f'git add output: {stdout + stderr}')
    
    stdout, stderr, _ = run_subprocess(["bash", "-c", "git diff --cached"])
    logs.append(f'git diff output: {stdout + stderr}')
    
    return stdout if stdout else "", logs

# =============================================================================
# Tool collection & schemas
# =============================================================================

TOOLS = [
    read_file, list_files, search_codebase, search_in_file, edit_file, edit_file_regex,
    insert_code_at_location, create_file, run_syntax_check, analyze_code_structure,
    discover_relevant_files, get_changes, finish
]

@lru_cache(maxsize=1)
def get_tool_schemas():
    """Generate tool schemas (cached)."""
    tool_schemas = []
    tool_name_to_args = {}
    
    for fn in TOOLS:
        name = fn.__name__
        doc = fn.__doc__ or ""
        sig = inspect.signature(fn)
        
        properties = {}
        required = []
        
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
                
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            
            # Simplified type mapping
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            json_type = "integer" if 'int' in type_hint else "boolean" if 'bool' in type_hint else "string"
            
            properties[param.name] = {
                "type": json_type,
                "description": f"Parameter: {param.name}"
            }
        
        tool_schemas.append({
            "name": name,
            "description": doc.strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        })
        
        tool_name_to_args[name] = [p.name for p in sig.parameters.values() if p.name != 'self']
    
    return tool_schemas, tool_name_to_args

TOOLS_SPECS, TOOL_NAME_TO_ARGS_NAME = get_tool_schemas()

@lru_cache(maxsize=1)
def get_tool_docs() -> str:
    """Generate tool documentation (cached)."""
    docs = []
    for i, tool in enumerate(TOOLS, 1):
        name = tool.__name__
        doc = inspect.getdoc(tool) or "Tool for code operations"
        sig = inspect.signature(tool)
        
        params = []
        for param_name, param in sig.parameters.items():
            if param.default != param.empty:
                default_val = "None" if param.default is None else str(param.default)
                params.append(f"{param_name}={default_val}")
            else:
                params.append(param_name)
        
        sig_str = f"{name}({', '.join(params)})"
        docs.append(f"{i}. {sig_str}\n   Description: {doc}")
    
    return '\n\n'.join(docs)

# =============================================================================
# Inference helpers
# =============================================================================

def _make_request(url: str, payload: dict) -> requests.Response:
    """Make HTTP request with timeout."""
    return requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)

def _parse_response(resp: requests.Response) -> str:
    """Parse inference response from various formats."""
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()
    text = resp.text or ""

    if "application/json" in content_type:
        try:
            data = resp.json()
            # Handle various response formats
            if isinstance(data, dict):
                # OpenAI-style chat completion
                if "choices" in data and data["choices"]:
                    choice = data["choices"][0]
                    if "message" in choice and isinstance(choice["message"], dict):
                        return choice["message"].get("content", "").strip()
                    if "text" in choice:
                        return choice["text"].strip()
                # Direct content field
                if "content" in data:
                    return data["content"].strip()
            # String response
            if isinstance(data, str):
                return data.strip()
        except json.JSONDecodeError:
            pass
    
    return text.strip()

def _request_with_retry(request_data: dict, url_base: str) -> str:
    """Make inference request with retry logic."""
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        try:
            # Ensure model is set
            request_data = {**request_data, 'model': request_data.get('model') or AGENT_MODELS[0]}
            
            # Try legacy endpoint first
            try:
                print(f" Trying legacy endpoint (attempt {attempt+1}/{MAX_RETRIES})...")
                url = f"{url_base.rstrip('/')}/agents/inference"
                resp = _make_request(url, request_data)
                content = _parse_response(resp)
                if content:
                    return content
                raise RuntimeError("Empty response from legacy endpoint")
            except Exception as e_legacy:
                last_error = e_legacy
                print(f" Legacy failed: {e_legacy}")
                
                # Try chat fallback if enabled
                if USE_CHAT_FALLBACK:
                    try:
                        print(" Trying chat fallback...")
                        url = f"{url_base.rstrip('/')}/chat/completions"
                        chat_payload = {
                            "model": request_data.get("model"),
                            "messages": request_data.get("messages", []),
                            "temperature": request_data.get("temperature", 0.0),
                            "stream": False,
                            "max_tokens": 2048,
                        }
                        resp = _make_request(url, chat_payload)
                        content = _parse_response(resp)
                        if content:
                            return content
                    except Exception as e_chat:
                        print(f" Chat fallback failed: {e_chat}")
                        last_error = e_chat
                
                # Wait before retry
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE * (2 ** attempt))  # Exponential backoff
                    
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE * (2 ** attempt))
    
    raise RuntimeError(f"Inference failed after {MAX_RETRIES} attempts: {last_error}")

def inference(messages: List[Dict[str, Any]], run_id: str) -> str:
    """Make inference call to the proxy/provider."""
    # Clean and validate messages
    cleaned_msgs = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") in {"system", "user", "assistant"} and str(m.get("content", "")).strip()
    ]

    request_data = {
        "run_id": run_id,
        "messages": cleaned_msgs,
        "temperature": 0.0,
        "model": AGENT_MODELS[0],
    }

    return _request_with_retry(request_data, DEFAULT_PROXY_URL)

# =============================================================================
# Action parsing
# =============================================================================

def extract_parameters(operation_name: str, param_string: str) -> dict:
    """Extract parameters with multiple parsing strategies."""
    # strip code fences if present
    param_string = re.sub(r'```(?:json)?\s*|\s*```', '', param_string).strip()
    
    # Strategy 1: JSON parsing
    try:
        result = json.loads(param_string)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    
    # Strategy 2: Safe eval
    try:
        result = eval(param_string, {"__builtins__": {}}, {})
        if isinstance(result, dict):
            return result
    except (SyntaxError, ValueError, NameError):
        pass
    
    # Strategy 3: Regex parsing
    if operation_name not in TOOL_NAME_TO_ARGS_NAME:
        raise ValueError(f"Unknown operation: {operation_name}")
    
    result = {}
    expected_keys = TOOL_NAME_TO_ARGS_NAME[operation_name]
    
    # Try quoted values first, then unquoted
    for key in expected_keys:
        # support both "key": "value" and 'key': 'value'
        patterns = [
            f'"{key}"\\s*:\\s*"([^"]*)"',
            f"'{key}'\\s*:\\s*'([^']*)'",
            f'"{key}"\\s*:\\s*([^,}}]+)',
            f"'{key}'\\s*:\\s*([^,}}]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, param_string)
            if match:
                result[key] = match.group(1).strip()
                break
    
    if not result:
        raise ValueError(f"Could not parse parameters: {param_string}")
    
    return result

def extract_action_format(text_resp: str) -> Tuple[str, str, dict]:
    """Extract structured action format from response."""
    text_resp = text_resp.strip()
    
    # Validate required fields
    required_fields = ['next_thought:', 'next_tool_name:', 'next_tool_args:']
    if not all(field in text_resp for field in required_fields):
        missing = [field for field in required_fields if field not in text_resp]
        raise ValueError(f"Missing required fields: {missing}")
    
    # Extract using regex
    pattern = r"next_thought:\s*(.*?)\s*next_tool_name:\s*(.*?)\s*next_tool_args:\s*(.*?)(?:\n\n|\Z)"
    match = re.search(pattern, text_resp, re.DOTALL)
    
    if not match:
        raise ValueError(f"Could not parse structured format: {text_resp[:200]}...")
    
    thought = match.group(1).strip()
    tool_name = match.group(2).strip().split('\n')[0].strip().strip('"\'')
    args_str = match.group(3).strip()
    
    tool_args = extract_parameters(tool_name, args_str)
    
    # Filter to expected arguments
    if tool_name in TOOL_NAME_TO_ARGS_NAME:
        expected_args = set(TOOL_NAME_TO_ARGS_NAME[tool_name])
        tool_args = {k: v for k, v in tool_args.items() if k in expected_args}
    
    return thought, tool_name, tool_args

def truncate_trajectory(trajectory: List[str], max_length: int = 15000) -> List[str]:
    """Truncate trajectory to prevent context overflow."""
    half_length = max_length // 2
    
    def truncate_text(text: str) -> str:
        if len(text) <= max_length:
            return text
        return f"{text[:half_length]}\n ... \n{text[-half_length:]}"
    
    return [truncate_text(item) for item in trajectory]

# =============================================================================
# Tool execution helpers (with auto-syntax-check + dedupe)
# =============================================================================

MODIFYING_TOOLS = {"edit_file", "edit_file_regex", "insert_code_at_location", "create_file"}

def _maybe_auto_syntax_check(tool_name: str, args: dict, observation: str) -> str:
    """If the tool modified a Python file, auto-run syntax check and append result."""
    try:
        if tool_name in MODIFYING_TOOLS:
            file_path = args.get("file_path")
            # Only run if it's a .py file and looks real
            if isinstance(file_path, str) and file_path.endswith(".py") and os.path.exists(file_path):
                syntax_obs = run_syntax_check(file_path)
                return f"{observation}\n\n{syntax_obs}"
    except Exception as e:
        return f"{observation}\n\n(Auto syntax-check failed: {e})"
    return observation

def execute_tool(tool_name: str, args: dict) -> str:
    """Execute a tool with given arguments."""
    tool_fn = next((tool for tool in TOOLS if tool.__name__ == tool_name), None)
    
    if tool_fn is None:
        return f"Error: Tool '{tool_name}' not found"
    
    try:
        observation = str(tool_fn(**args))
        observation = _maybe_auto_syntax_check(tool_name, args, observation)
        return observation
    except Exception as e:
        return f"Error executing {tool_name}: {e}"

# =============================================================================
# Main workflow
# =============================================================================

def execute_workflow(problem_statement: str, timeout: int, run_id: str, instance_id: str = "") -> Tuple[str, List[str], List[str]]:
    """Execute the workflow with timeout and trajectory management."""
    logs = [f"Working directory: {os.getcwd()}"]
    start_time = time.time()
    
    # Clean up potential file conflicts
    cleanup_files = ['./src/agent.py', './agent_runner.py']
    for file in cleanup_files:
        try:
            os.remove(file)
            print(f"Deleted {file}")
        except OSError:
            pass
    
    # Create prompts
    tool_docs = get_tool_docs()
    system_prompt = PROMPTS["system"].format(tools_docs=tool_docs, format_prompt=PROMPTS["format"])
    instance_prompt = PROMPTS["instance"].format(problem_statement=problem_statement)
    
    trajectory = [problem_statement]
    consecutive_errors = 0

    # Track duplicate tool calls (tool_name + sorted args) to avoid wasted steps
    seen_calls = set()

    print("🚀 Starting optimized agent...")
    
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step + 1} ---")
        
        # Time management
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        
        if remaining < 50:  # Hard timeout
            print("⏰ Workflow timeout reached")
            logs.append(f"Workflow timeout at step {step + 1}")
            break
            
        if remaining < (REQUEST_TIMEOUT + 10):  # Not enough time for safe request
            print(f"⏳ Insufficient time remaining ({int(remaining)}s); ending workflow.")
            logs.append(f"Insufficient time at step {step + 1}")
            break
        
        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instance_prompt},
            {"role": "user", "content": '\n'.join(trajectory)},
            {"role": "system", "content": PROMPTS["stop"]},
            {"role": "user", "content": f"Execute next action for: {problem_statement}"}
        ]
        
        trajectory = truncate_trajectory(trajectory)
        
        # Get LLM response
        try:
            text_resp = inference(messages, run_id)
            consecutive_errors = 0
            print(f"Response: {text_resp[:200]}...")
        except Exception as e:
            consecutive_errors += 1
            error_msg = f"Inference error: {e}"
            print(error_msg)
            trajectory.append(error_msg)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"🚫 Too many consecutive errors ({consecutive_errors}); aborting.")
                break
            continue
        
        # Parse action
        try:
            thought, tool_name, tool_args = extract_action_format(text_resp)
            print(f"Tool: {tool_name}, Args: {tool_args}")
        except Exception as e:
            error_msg = f"Parse error: {e}"
            print(error_msg)
            trajectory.append(error_msg)
            continue
        
        # Deduplicate exact same tool call + args to save steps
        try:
            fingerprint = (tool_name, tuple(sorted(tool_args.items())))
            if fingerprint in seen_calls:
                skip_msg = f"Skipping duplicate tool call: {tool_name} {tool_args}"
                print(skip_msg)
                trajectory.append(skip_msg)
                continue
            seen_calls.add(fingerprint)
        except Exception:
            # If anything odd in args, just proceed
            pass
        
        # Execute tool
        try:
            observation = execute_tool(tool_name, tool_args)
            trajectory.extend([
                f"next_thought: {thought}",
                f"next_tool_name: {tool_name}",
                f"next_tool_args: {tool_args}",
                f"observation: {observation}"
            ])
            print(f"Observation: {observation[:200]}...")
        except Exception as e:
            error_msg = f"Tool execution error: {e}"
            print(error_msg)
            trajectory.append(error_msg)
            continue
        
        if tool_name == "finish":
            break
    
    # Get final patch
    try:
        patch, patch_logs = get_final_git_patch()
        logs.extend(patch_logs)
    except Exception as e:
        logs.append(f"Error getting patch: {e}")
        patch = ""
    
    logs.extend(trajectory)
    return patch, trajectory, logs

def agent_main(input_dict: Dict[str, Any]) -> Dict[str, str]:
    """Main entry point for the agent."""
    problem_statement = input_dict.get("problem_statement")
    if not problem_statement:
        raise ValueError("Missing 'problem_statement' in input")
    
    run_id = input_dict.get("run_id", "default")
    instance_id = input_dict.get("instance_id", "default")
    
    # Change to repo directory if it exists
    if os.path.exists("repo"):
        os.chdir("repo")
    
    try:
        # Reset git state
        subprocess.run(["git", "reset", "--hard"], capture_output=True)
        subprocess.run(["git", "config", "--global", "--add", "safe.directory", "/sandbox/repo"], capture_output=True)
        
        # Execute workflow
        timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
        patch, trajectory, logs = execute_workflow(problem_statement, timeout, run_id, instance_id)
        
        # Reset again (harness applies patch; keep working copy clean)
        subprocess.run(["git", "reset", "--hard"], capture_output=True)
        
        return {"patch": patch}
        
    except Exception as e:
        error_msg = f"Agent error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return {"patch": f"Error: {error_msg}"}
