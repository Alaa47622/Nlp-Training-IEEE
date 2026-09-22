install:
	uv sync

run:
	uv run streamlit run app/main.py

test:
	uv run pytest

docker-build:
	docker build -t toxic-content-detector .

docker-run:
	docker run --env-file .env -p 8501:8501 toxic-content-detector
