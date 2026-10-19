Getting started
===============

The README contains a comprehensive overview of Hiyal's capabilities, installation
steps, and quickstart examples. The most common workflows include:

* Creating agents via :class:`hiyal.agents.AgentManager`.
* Scheduling coordinated interactions with :class:`hiyal.schedulers.RoundRobinScheduler`
  or any of the registry-backed schedulers in :mod:`hiyal.schedulers`.
* Capturing environment state and telemetry using :class:`hiyal.environments.Environment`
  and :mod:`hiyal.utils.observability.telemetry` helpers.

For local development, ensure the development dependencies are installed and run the
commands listed in the ``Testing & Quality Assurance`` section of the README to execute
unit tests, integration tests, linters, and documentation builds.
