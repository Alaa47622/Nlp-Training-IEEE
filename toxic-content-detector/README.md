# Toxic Content Detection 


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

to the Docker image and `.env` is excluded by `.gitignore`
