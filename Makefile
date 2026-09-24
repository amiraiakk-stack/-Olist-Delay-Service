install:
	pip install -r requirements-dev.txt

fixture-artifacts:
	python -m scripts.build_fixture_artifacts

test:
	pytest -q

lint:
	ruff check app src tests
	ruff format --check app src tests

run:
	uvicorn app.main:app --reload

compose-up:
	docker compose up --build
