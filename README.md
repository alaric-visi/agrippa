# agrippa
An AI agent designed to solve SWE-Bench coding challenges, using the RidgesAI framework.

## What Ridges Is
Ridges is an open‑source, decentralised platform designed to replace human software engineers with competing AI agents. It operates on the Bittensor blockchain and offers a winner‑take‑all competition where agents are evaluated on real software engineering tasks. 

The core idea is to incentivise individuals to build autonomous code‑solving agents by paying daily rewards. The Ridges overview notes that the network is a distributed AI evaluation platform that uses blockchain incentives to encourage development of autonomous code‑solving agents. 

Agents are evaluated on real-world software engineering benchmarks (SWE‑bench) using a multi‑component architecture and the best performers earn rewards.

- https://www.ridges.ai/
- https://docs.ridges.ai/
- https://www.swebench.com/
- https://bittensor.com/
- https://chutes.ai/

### The platform has several cooperating components:
- **Platform**: the central coordination service. It manages agent submissions, assigns jobs to validators, and provides REST and WebSocket APIs.
- **Proxy**: a secure gateway between agents and external AI providers (Chutes AI). It validates requests via run‑IDs, enforces per‑run cost limits, and forwards approved inference or embedding calls. The proxy ensures that only requests from valid evaluation runs are processed and tracks the cost of each run.
- **Screeners**: preliminary evaluators that run light tests on new agents to filter out weak submissions before full evaluation.
- **Validators**: distributed evaluation nodes that run agents on SWE‑bench problems in sandboxed Docker containers. They independently evaluate each agent’s patch, aggregate results and contribute to consensus scoring on the blockchain.
- **Miners and agents**: developers (miners) who write and submit agents. Miners compete to produce the highest‑scoring agent and are rewarded accordingly.

Agents submitted to the platform follow an evaluation pipeline. They are first screened on five easy tasks; only agents solving enough (currently 3 of 5) progress to full evaluation. 

Validators then run each surviving agent on ~50 SWE‑bench problems, checking that the generated patches apply cleanly and pass tests. Scores from multiple validators are aggregated to determine the top agent. 

Agents run in isolated containers and cannot make arbitrary external calls. Instead, each agent receives environment variables specifying a proxy URL and a timeout; external requests must go through the proxy and are limited by a cost cap.

### Requirements for Agents
Ridges imposes strict requirements on submitted agents. The Miner Guide explains that an agent must be a single Python file containing a top‑level function called agent_main. This function receives an input_dict containing at least a problem_statement and returns a dictionary with a patch key holding a git diff. 

The agent can only use Python’s standard library and a limited set of approved external libraries. 

Agents are sandboxed under the /repo path and cannot access external systems directly; all external inference or embedding requests must go through the proxy using the AI_PROXY_URL environment variable. The sandbox enforces timeouts (currently two minutes) and cost caps on inference and embeddings.

## My agent
The Python implements an autonomous software‑engineering agent conforming to the Ridges requirements. It acts as the “brain” that reads a problem statement, explores the repository, generates code edits, and returns a patch. The code is structured into several sections.

### Configuration and prompts
The file defines constants for environment variables such as AI_PROXY_URL, AGENT_TIMEOUT, and the model identifier used for inference (AGENT_MODELS). It sets tunable limits such as MAX_STEPS (maximum actions per problem), MAX_RETRIES, and MAX_CONSECUTIVE_ERRORS. 

A PROMPTS dictionary contains formatted strings used to instruct the large language model (LLM). 

The system prompt tells the model that it acts as a senior Python engineer who should understand the problem, explore the codebase, split the task into small tasks, and make code changes using provided tools; it also reminds the model not to modify tests and to run syntax checks after edits.

### Utility functions and core tools
To allow the agent to inspect and modify the repository, the file provides a set of tools, each wrapped with a handle_errors decorator to capture exceptions and return human‑readable messages.

- ```read_file``` reads a file’s content, optionally with a line range, and truncates large files for efficiency.
- ```edit_file``` replaces a specific block of code in a file; it refuses to modify test files and ensures only one occurrence is replaced.
- ```edit_file_regex``` performs regex‑based substitutions.
- ```create_file creates``` a new file (non‑test).
- ```insert_code_at_location``` inserts code before or after a specific line number.
- ```list_files``` runs a find command to list Python files, excluding ```__pycache__``` and test directories.
- ```search_codebase``` and ```search_in_file``` use grep to locate occurrences of a search term.
- ```run_syntax_check``` compiles a Python file to check for syntax errors.
- ```analyze_code_structure``` uses the AST module to list imports, classes, functions and global variables in a file, and provides simple metrics such as number of lines.
- ```discover_relevant_files``` looks for files in the same directory or files that import or are imported by a given module.
- ```get_changes``` returns all code changes (using ```git diff```) and ```finish``` is used to signal completion.

hese tools correspond to the predefined API that Ridges exposes to agents within the sandbox. Agents cannot call arbitrary system commands; instead, they must call these tool functions to read or modify files. 

Restricting actions to these tools aligns with Ridges’ security and isolation principles: code runs in a sandbox and cannot modify tests or perform dangerous operations.

### Tool schema generation and documentation
To help the LLM choose which tool to call, the agent dynamically generates JSON schemas and human‑readable documentation for each tool. The function ```get_tool_schemas``` examines the function signatures of all tools and produces JSON metadata with parameter types. 

```get_tool_docs``` builds a string describing each tool’s signature and docstring. These are inserted into the system prompt so the LLM knows how to call the tools.


