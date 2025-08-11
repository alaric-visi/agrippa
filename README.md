# AGRIPPA
An AI agent designed to solve SWE-Bench coding challenges, using the RidgesAI framework.

## What Ridges is
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
Ridges imposes strict requirements on submitted agents. The Miner Guide explains that an agent must be a single Python file containing a top‑level function called ```agent_main```. This function receives an input_dict containing at least a problem_statement and returns a dictionary with a patch key holding a git diff. 

The agent can only use Python’s standard library and a limited set of approved external libraries. 

Agents are sandboxed under the /repo path and cannot access external systems directly; all external inference or embedding requests must go through the proxy using the ```AI_PROXY_URL``` environment variable. The sandbox enforces timeouts (currently two minutes) and cost caps on inference and embeddings.

## My agent
The Python implements an autonomous software‑engineering agent conforming to the Ridges requirements. It acts as the “brain” that reads a problem statement, explores the repository, generates code edits, and returns a patch. The code is structured into several sections.

### Configuration and prompts
The file defines constants for environment variables such as ```AI_PROXY_URL```, ```AGENT_TIMEOUT```, and the model identifier used for inference (AGENT_MODELS). It sets tunable limits such as ```MAX_STEPS``` (maximum actions per problem), ```MAX_RETRIES```, and ```MAX_CONSECUTIVE_ERRORS```. 

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

These tools correspond to the predefined API that Ridges exposes to agents within the sandbox. Agents cannot call arbitrary system commands; instead, they must call these tool functions to read or modify files. 

Restricting actions to these tools aligns with Ridges’ security and isolation principles: code runs in a sandbox and cannot modify tests or perform dangerous operations.

### Tool schema generation and documentation
To help the LLM choose which tool to call, the agent dynamically generates JSON schemas and human‑readable documentation for each tool. The function ```get_tool_schemas``` examines the function signatures of all tools and produces JSON metadata with parameter types. 

```get_tool_docs``` builds a string describing each tool’s signature and docstring. These are inserted into the system prompt so the LLM knows how to call the tools.

### Inference helpers
Agents in Ridges are not allowed to use arbitrary external services. Instead, they must call the proxy for inference. The functions ```_make_request```, ```_parse_response``` and ```_request_with_retry``` implement this logic. ```_make_request``` makes an ```HTTP POST``` to the proxy using the configured ```REQUEST_TIMEOUT```. ```_parse_response``` handles the different response formats from the proxy and returns the generated text. ```_request_with_retry``` attempts inference up to ```MAX_RETRIES```, first calling the legacy /agents/inference endpoint on the proxy and optionally falling back to a chat endpoint when enabled. It uses exponential backoff to respect rate limits.

The top‑level inference function wraps this logic: it builds a request with the cleaned message history, sets the desired model and temperature, and sends it through the proxy. Because the ```run_id``` is included in the request, the proxy can verify that the call is authorised and track cost; this matches the Ridges proxy’s requirement that each request includes a valid run ID and that cost is tracked per evaluation run.

### Action parsing
Once the LLM produces a response, the agent must parse it to extract the next action. The helper extract_action_format expects the LLM output to contain:
```
next_thought: …
next_tool_name: …
next_tool_args: …
```
It uses regex to extract these fields and then ```extract_parameters``` to parse the JSON‑like argument string, supporting JSON, Python literal syntax and fallback regex extraction. If the model omits required fields or provides invalid JSON, the agent reports an error.

### Tool execution and workflow control
The key orchestration happens in execute_workflow. After resetting the git state, it constructs the system prompt (including tool docs) and an instance prompt based on the problem statement. It maintains a trajectory list recording all previous steps. For each step (up to ```MAX_STEPS```), it:

1. Builds a message history containing the system prompt, problem statement, previous trajectory, a stop instruction, and a final user instruction requesting the next action.
2. Sends this to the LLM via inference and parses the response.
3. Deduplicates identical tool calls to avoid wasted steps.
4. Executes the chosen tool using execute_tool. If a modifying tool is used (edit, insert or create file), ```_maybe_auto_syntax_check``` automatically runs ```run_syntax_check``` on the modified file to catch syntax errors early. This mirrors the Ridges requirement that agents maintain valid syntax and avoid invalid patches.
5. Appends the observation (tool output) to the trajectory and repeats.

The loop stops if the ```finish``` tool is called, if a timeout occurs (```AGENT_TIMEOUT``` minus the time spent), or if too many consecutive errors are observed. Once the loop ends, execute_workflow collects a git diff of all changes (```get_final_git_patch```) and returns it along with logs.

### Entry point
The ```agent_main``` function is the required entry point for Ridges agents. It extracts the ```problem_statement```, ```run_id``` and ```instance_id``` from the input dictionary. It ensures that the working directory points to the repo (where the target project lives). 

It resets the git state to a clean slate, executes the workflow with the specified timeout, and returns a dictionary containing the final patch. Returning a patch in this format satisfies Ridges’ requirement that agents return a git diff through the patch key. 

After producing the patch, it resets the git state again so that the harness can apply the patch separately.

## How the agent fits into the Ridges framework
The agent implements the behaviours expected by the Ridges platform:

1. **Single‑file agent with agent_main** – The file defines an agent_main function at the top level that accepts an ```input_dict``` and returns a patch. This matches the agent structure requirements documented in the Miner Guide. Because the entire agent logic is contained in one file, it can be easily uploaded via the platform’s /upload endpoint.
2. **Use of Ridges’ proxy for inference** – External model calls go through ```_request_with_retry```, which sends ```POST``` requests to ```DEFAULT_PROXY_URL``` (derived from ```AI_PROXY_URL```). The proxy validates each request by run ID and enforces cost limits, as described in the Proxy documentation. This ensures the agent respects per‑run cost caps and cannot bypass cost control.
3. **Respect for sandbox restrictions** – The agent interacts with the repository exclusively through pre‑approved tools (read, search, edit, etc.) and refuses to modify test files. This aligns with the platform’s security rules (sandboxed execution, code validation and import restrictions). The agent never executes arbitrary shell commands except through controlled functions like run_subprocess inside certain tools, and even then only for search or listing files.
4. **Automatic syntax checking** – After each file modification, the agent runs run_syntax_check. Ridges validators judge patches based on whether they compile and pass tests; by checking syntax early, the agent reduces the risk of submitting invalid patches. Validators run each agent’s code in isolated Docker containers and test patches on SWE‑bench problems, so syntax errors would cause failures.
5. **Trajectory management and timeouts** – The workflow enforces a maximum number of steps and stops early if insufficient time remains. The miner guide notes that agents have limited time and cost budgets in the sandbox. By truncating trajectories, deduplicating calls and respecting timeouts, the agent uses its allowed resources efficiently.
6. **Returning a patch** – Finally, ```agent_main returns``` a dictionary with a patch key containing a git diff of the changes. Validators will apply this patch to the repository and run tests. This output format matches the Ridges specification.

Overall, this Python file implements a self‑contained software‑engineering agent designed to operate within the Ridges AI ecosystem. It leverages the proxy for inference, uses sandbox‑approved tools to inspect and modify code, enforces syntax correctness, and produces a patch that can be evaluated by validators. Its design directly mirrors Ridges’ architecture and evaluation process, ensuring compatibility with the competition’s rules and allowing it to participate in the Ridges marketplace of autonomous agents.
