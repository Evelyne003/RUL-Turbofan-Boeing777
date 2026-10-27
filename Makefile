# Trabajo Terminal 2027-A190 — Estimación de RUL en motores turbofan (C-MAPSS)
PYTHON ?= python
SUBSET ?= FD001
WINDOW ?= 30
EPOCHS ?= 100

.PHONY: help setup data train train-all grid evaluate dashboard test lint clean

help:
	@echo "make setup       - instala dependencias de desarrollo"
	@echo "make data        - descarga C-MAPSS en data/raw (ver README si falla la URL)"
	@echo "make train       - entrena un subconjunto: make train SUBSET=FD002 WINDOW=20"
	@echo "make train-all   - entrena FD001..FD004"
	@echo "make grid        - búsqueda en cuadrícula (selección por validación): make grid SUBSET=FD001"
	@echo "make evaluate    - reevalúa un modelo guardado sobre el conjunto de prueba"
	@echo "make dashboard   - abre el tablero Streamlit"
	@echo "make test        - ejecuta pytest"
	@echo "make lint        - ruff"

setup:
	$(PYTHON) -m pip install -r requirements-dev.txt

data:
	$(PYTHON) -m src.ingestion --download

train:
	$(PYTHON) -m src.train --subset $(SUBSET) --window $(WINDOW) --epochs $(EPOCHS)

train-all:
	$(PYTHON) -m src.train --subset FD001 --window 30 --epochs $(EPOCHS)
	$(PYTHON) -m src.train --subset FD002 --window 20 --epochs $(EPOCHS)
	$(PYTHON) -m src.train --subset FD003 --window 30 --epochs $(EPOCHS)
	$(PYTHON) -m src.train --subset FD004 --window 20 --epochs $(EPOCHS)

grid:
	$(PYTHON) -m src.grid_search --subset $(SUBSET) --epochs $(EPOCHS)

evaluate:
	$(PYTHON) -m src.evaluation --subset $(SUBSET)

dashboard:
	streamlit run dashboard/app.py

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests dashboard

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
