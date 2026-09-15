from pathlib import Path


def test_streamlit_app_exists():
    assert Path("app.py").exists()


def test_ui_dockerfile_exists():
    dockerfile = Path("Dockerfile.ui")
    assert dockerfile.exists()
    content = dockerfile.read_text(encoding="utf-8")
    assert "streamlit" in content.lower()
    assert "8501" in content


def test_compose_exposes_streamlit_port():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    assert "8501:8501" in compose
