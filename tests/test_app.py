from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_views_and_unseen_hawaii_input(monkeypatch):
    monkeypatch.delenv("SALARY_ENABLE_MLFLOW", raising=False)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py"), default_timeout=30).run()
    assert not app.exception
    for page in ["Model comparison", "Feature sensitivity"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    app.sidebar.radio[0].set_value("Model comparison").run()
    next(item for item in app.selectbox if item.label == "job_state").set_value("HI")
    app.button[0].click().run()
    assert not app.exception
    assert len(app.metric) == 2
    assert any("absent from training" in warning.value for warning in app.warning)
