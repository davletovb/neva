# Neva: Creating Multi-Agent Simulations with Large Language Models!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Issues](https://img.shields.io/github/issues/davletovb/neva)](https://github.com/davletovb/neva/issues)
[![Last Commit](https://img.shields.io/github/last-commit/davletovb/neva)](https://github.com/davletovb/neva/commits/master)
[![Contributors](https://img.shields.io/github/contributors/davletovb/neva)](https://github.com/davletovb/neva/graphs/contributors)
[![Forks](https://img.shields.io/github/forks/davletovb/neva?style=social)](https://github.com/davletovb/neva/fork)
[![Stars](https://img.shields.io/github/stars/davletovb/neva?style=social)](https://github.com/davletovb/neva/stargazers)

<p align="center">
  <img src="https://github.com/davletovb/neva/assets/43503037/e9a2627b-e328-4986-a669-9ac13ad438b4" alt="Intellibot Logo">
</p>

Want to create worlds where AI agents come alive? 🤖 Our open-source library lets you easily build captivating simulations where customizable agents interact using natural language.💬 Immerse yourself in emergent behaviors as your intelligent assistants, characters, and communities cooperate in exciting new ways! 🔥 Watch the dynamics unfold as your agents chat, explore, and work together powered by state-of-the-art LLMs. 💡 Join our open-source movement to make agent-based modeling more accessible, from classrooms to cutting-edge research. 🧑‍🏫👩‍🔬  

**We seek Python developers passionate about integrating LLMs to collaborate on an open-source library for modeling complex emergent behaviors and human-AI cooperation dynamics.**

## Table of Contents
- [Why Neva?](#why-neva)
- [Quickstart 🚀](#quickstart-)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Get Involved 🤝](#get-involved-)
- [License](#license)

## Why Neva?
Neva makes it simple to unlock the power of large language models through specialized AI agents. Benefits include:

- **Rapid Prototyping** - Quickly build conversational agents, productivity tools, game characters, and more.
- **Modular Components** - Switch between different types of LLMs. Mix and match configurable agents, tools, and environments.
- **Scalable Systems** - Develop complex ecosystems of hierarchical, coordinating agents.
- **Simulation Capabilities** - Model and evaluate agent behaviors under various conditions.
- **Community Resources** - Leverage shared agents, tools, and examples from our open-source community.

Neva empowers developers, researchers, educators, and hobbyists to create the next generation of AI interactions for gaming, automation, education, research, and beyond!

## Quickstart 🚀
Dive right into Neva with our easy and intuitive quickstart guide. Whether you're a seasoned AI enthusiast or just exploring, you can access our collection of specialized AI agents, tools, and environments. Simply select the components you need, and customize them to your liking (like defining specific tasks or integrating with your favorite tools). With just a few clicks, you'll be ready to unleash the power of AI!

Use our handy code snippets to integrate Neva into your Python projects, GitHub repositories, or other platforms. It's that simple!

## Installation
Clone the repository and install the required packages:

```bash
git clone https://github.com/davletovb/neva.git
cd neva
pip install -r requirements.txt
```

The dependencies include libraries such as `openai`, `transformers`, `wikipedia`, `googletrans`, and `bert-extractive-summarizer`.

## Usage
Create and simulate AI agents effortlessly:
```python
from environments import BasicEnvironment
from models import AgentManager
from schedulers import RoundRobinScheduler
from tools import MathTool

# 1. Create an agent using the manager
manager = AgentManager()
agent = manager.create_agent(
    "transformer",
    name="Tutor",
    llm_backend=lambda prompt: f"(stubbed model) {prompt}",
)
agent.register_tool(MathTool())

# 2. Prepare an environment and scheduler
scheduler = RoundRobinScheduler()
environment = BasicEnvironment("Classroom", "A simple maths lesson", scheduler)
environment.register_agent(agent)

# 3. Drive the simulation loop
for _ in range(3):
    response = environment.step()
    print(response)
```

Each call to ``environment.step`` asks the scheduler for an agent, collects
metrics via the observer system, and invokes the agent's ``respond`` method with
the current environmental context.  This mirrors the simulation lifecycle used
throughout the library and in the accompanying tests.

## Features
- **Flexible & Adaptable**: Adapt to various types of LLMs, tasks, and tools.
- **Hierarchical Agents**: Design agents for specific tasks and coordinate them harmoniously.
- **Intuitive Interfaces**: Simple interfaces for agent creation and management.
- **Environment Simulation**: Simulate environments for agent interactions and collaborations.

## Get Involved 🤝

Neva thrives on community collaboration, and we welcome your participation through issues and pull requests! Whether you're fixing a bug, proposing a new feature, or enhancing our documentation, your contributions are the heartbeat of our project.

Before jumping in, take a moment to read our [contribution guidelines.](https://github.com/davletovb/neva/blob/main/CONTRIBUTING.md)

- **Report bugs** by opening a GitHub issue.
- **Suggest enhancements** through the issues tracker.
- **Improve documentation** with pull requests.
- **Share examples** of Neva projects and use cases.
- **Add tests** to improve coverage.

We value each contribution, no matter how big or small, and we're excited to see what you'll bring to our growing community!

## License
This project is licensed under the MIT License. See the [License File](https://github.com/davletovb/neva/blob/main/LICENSE) for more details.
