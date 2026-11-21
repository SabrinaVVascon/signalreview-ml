from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_rejects_short_text():
    response = client.post("/predict", json={"texto": "a"})
    assert response.status_code == 422


def test_predict_without_training_or_with_training():
    response = client.post("/predict", json={"texto": "O produto chegou no prazo e funcionou bem."})
    # Sem modelo ainda: 503; apos treino: 200.
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        body = response.json()
        assert body["sentimento"] in ("positivo", "negativo")
        assert 0 <= body["confianca"] <= 1
