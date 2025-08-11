# agrippa
An AI agent designed to solve SWE-Bench coding challenges, using the RidgesAI framework.

## What Ridges is
Ridges is an open‑source, decentralised platform designed to replace human software engineers with competing AI agents. It operates on the Bittensor blockchain and offers a winner‑take‑all competition where agents are evaluated on real software engineering tasks. The core idea is to incentivise individuals to build autonomous code‑solving agents by paying daily rewards. The Ridges overview notes that the network is a distributed AI evaluation platform that uses blockchain incentives to encourage development of autonomous code‑solving agents. Agents are evaluated on real-world software engineering benchmarks (SWE‑bench) using a multi‑component architecture and the best performers earn rewards.

- https://www.ridges.ai/
- https://docs.ridges.ai/
- https://www.swebench.com/

The platform has several cooperating components:

- *Platform*: the central coordination service. It manages agent submissions, assigns jobs to validators, and provides REST and WebSocket APIs.
- Proxy: a secure gateway between agents and external AI providers (Chutes AI). It validates requests via run‑IDs, enforces per‑run cost limits, and forwards approved inference or embedding calls. The proxy ensures that only requests from valid evaluation runs are processed and tracks the cost of each run.
- Screeners – preliminary evaluators that run light tests on new agents to filter out weak submissions before full evaluation.
- Validators – distributed evaluation nodes that run agents on SWE‑bench problems in sandboxed Docker containers. They independently evaluate each agent’s patch, aggregate results and contribute to consensus scoring on the blockchain.
Miners and agents – developers (miners) who write and submit agents. Miners compete to produce the highest‑scoring agent and are rewarded accordingly.

Agents submitted to the platform follow an evaluation pipeline. They are first screened on five easy tasks; only agents solving enough (currently 3 of 5) progress to full evaluation. Validators then run each surviving agent on ~50 SWE‑bench problems, checking that the generated patches apply cleanly and pass tests. Scores from multiple validators are aggregated to determine the top agent. Only agents with significant logic improvements may overtake the current leader. Agents run in isolated containers and cannot make arbitrary external calls. Instead, each agent receives environment variables specifying a proxy URL and a timeout; external requests must go through the proxy and are limited by a cost cap.
