# LangChain & LangGraph Agent Orchestration

This repository demonstrates advanced orchestration techniques using LangChain and LangGraph frameworks to create a system of specialized agent nodes working together.

## Overview

This project implements an AI workflow orchestration system where multiple specialized agents collaborate to solve complex tasks. The system uses:

- **LangChain** for the core chains and agent functionality
- **LangGraph** for workflow orchestration and state management
- **OpenAI's models** as the underlying LLM technology

The architecture follows a modular approach with specialized agents that each handle different aspects of problem-solving:

1. **Research Agent** - Gathers relevant information on the query
2. **Coding Agent** - Generates Python code based on research findings
3. **Summary Agent** - Creates concise summaries of research and code
4. **Decision Agent** - Determines the next workflow step
5. **Final Agent** - Produces comprehensive answers integrating all information
