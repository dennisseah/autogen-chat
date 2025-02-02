# autogen-chat
AutoGen Agent Chat

## Setup

1. Clone the repository
2. `cd autogen-chat` (root directory of this git repository)
3. `python -m venv .venv`
4. `poetry install` (install the dependencies)
5. code . (open the project in vscode)
6. install the recommended extensions (cmd + shift + p -> `Extensions: Show Recommended Extensions`)
7. `pre-commit install` (install the pre-commit hooks)
8. copy the `.env.sample` file to `.env` and fill in the values

## Samples
```sh
python -m autogen_chat.main
```

## Unit Test Coverage

```sh
python -m pytest -p no:warnings --cov-report term-missing --cov=autogen_chat tests
```
