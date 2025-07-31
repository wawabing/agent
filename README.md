# Agent Upload Project

This project contains machine learning agents for anomaly detection and root cause analysis.

## Files

- `agent_detector.py` - Main agent detection functionality
- `agent_rca.py` - Root cause analysis agent
- `langchain_pipeline.py` - LangChain integration pipeline
- `memory/` - Directory containing memory files for agents
- `results/` - Directory for storing analysis results

## Setup

1. Install required dependencies:
```bash
pip install pandas openai scikit-learn scipy matplotlib python-dotenv tiktoken langgraph
```

2. Set up your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Run the main detection agent:
```bash
python agent_detector.py
```

Run root cause analysis:
```bash
python agent_rca.py
```

## Requirements

- Python 3.7+
- OpenAI API key
- Required Python packages (see setup above)
