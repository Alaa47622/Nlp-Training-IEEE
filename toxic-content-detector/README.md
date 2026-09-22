# Toxic Content Detection — Task 2 + Task 3

A beginner-friendly Toxic Content Detection application built with **LangChain, OpenRouter, Streamlit, uv, and Docker**.

## Assignment coverage

### Task 2 — LangChain + OpenRouter + Streamlit
- Accept user-provided text.
- Send the text through a LangChain prompt.
- Use an LLM through OpenRouter.
- Classify the input as **Toxic** or **Non-Toxic**.
- Show a short explanation.
- Provide a simple Streamlit web interface.

### Task 3 — Docker
- Dockerize the application.
- Install dependencies using **uv**.
- Keep the OpenRouter API key outside the image.
- Expose Streamlit on port `8501`.
- Run the application inside a Docker container.

## Where is uv?

`uv` is the Python project/dependency manager for this project. It is configured in `pyproject.toml` and the project pins Python 3.12 in `.python-version`.

Use:

```powershell
uv sync
uv run streamlit run app/main.py
```

After running `uv sync` on a machine with internet access, uv will generate `uv.lock`. The supplied Dockerfile also installs uv inside the container.

## Where is the data file?

This application **does not train a machine-learning model**, so a training dataset is not required. The LLM performs the classification at runtime.

For the assignment and testing, a small sample dataset is included at:

```text
data/sample_data.csv
```

It contains example texts and expected `Toxic` / `Non-Toxic` labels that can be used for manual testing or future automated evaluation.

## Project structure

```text
toxic-content-detector/
├── app/
│   ├── __init__.py
│   ├── classifier.py
│   └── main.py
├── data/
│   └── sample_data.csv
├── screenshots/
│   └── README.md
├── tests/
│   └── test_classifier.py
├── .dockerignore
├── .env.example
├── .gitignore
├── .python-version
├── Dockerfile
├── Makefile
├── README.md
└── pyproject.toml
```

## 1. Requirements

Install:

- Python 3.12
- uv
- Docker Desktop
- An OpenRouter API key

Check:

```powershell
python --version
uv --version
docker --version
```

## 2. Setup with uv

From the project root:

```powershell
uv sync
```

This creates the project virtual environment and installs the dependencies declared in `pyproject.toml`.

## 3. Configure OpenRouter

Copy `.env.example` to `.env` and add your key:

```env
OPENROUTER_API_KEY=your_real_key_here
OPENROUTER_MODEL=openai/gpt-oss-20b
```

Never commit `.env` to GitHub.

## 4. Run locally

```powershell
uv run streamlit run app/main.py
```

Open:

```text
http://localhost:8501
```

## 5. Run tests

```powershell
uv run pytest
```

## 6. Docker build

```powershell
docker build -t toxic-content-detector .
```

## 7. Docker run

```powershell
docker run --env-file .env -p 8501:8501 toxic-content-detector
```

Open:

```text
http://localhost:8501
```

## 8. Docker verification

The expected flow is:

```text
Browser
   |
   | http://localhost:8501
   v
Docker Container
   |
   v
Streamlit
   |
   v
LangChain
   |
   v
OpenRouter
   |
   v
LLM
   |
   v
Toxic / Non-Toxic + explanation
```

## 9. Screenshots for submission

Save your real screenshots in the `screenshots/` folder.

### Screenshot 1 — Project structure

Show VS Code with:
- `app/`
- `data/`
- `Dockerfile`
- `pyproject.toml`
- `uv.lock` after running `uv sync`

### Screenshot 2 — Streamlit UI

Show the browser with the Toxic Content Detector interface.

### Screenshot 3 — Non-Toxic result

Use an example such as:

```text
I really enjoyed this movie.
```

Show the **Non-Toxic** classification and explanation.

### Screenshot 4 — Toxic result

Use an example from `data/sample_data.csv` and show the **Toxic** classification.

### Screenshot 5 — Docker build

Show:

```text
docker build -t toxic-content-detector .
```

ending successfully.

### Screenshot 6 — Docker container

Show:

```text
docker run --env-file .env -p 8501:8501 toxic-content-detector
```

### Screenshot 7 — Browser through Docker

Show the Streamlit UI at:

```text
http://localhost:8501
```

while the Docker container is running.

## Security note

The OpenRouter API key is supplied at runtime using `.env`. It is not copied into the Docker image and `.env` is excluded by `.gitignore`.
