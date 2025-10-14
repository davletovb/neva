Getting started
===============

The README contains a comprehensive overview of Neva's capabilities, installation
steps, and quickstart examples. The most common workflows include:

* Creating agents via :class:`neva.agents.AgentManager`.
* Scheduling coordinated interactions with :class:`neva.schedulers.RoundRobinScheduler`
  or any of the registry-backed schedulers in :mod:`neva.schedulers`.
* Capturing environment state and telemetry using :class:`neva.environments.Environment`
  and :mod:`neva.utils.telemetry` helpers.

For local development, ensure the development dependencies are installed and run the
commands listed in the ``Testing & Quality Assurance`` section of the README to execute
unit tests, integration tests, linters, and documentation builds.
