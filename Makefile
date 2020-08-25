PYTHON=.venv/Scripts/python.exe
PIP=.venv/Scripts/pip.exe
FLAKE8=.venv/Scripts/flake8.exe

init:
	python -m venv .venv
	.venv\scripts\activate

dev:
	${PYTHON} -m pip install --upgrade pip
	${PIP} install -r dev.requirements.txt

build:
    ${PYTHON} -m pip install --upgrade pip
	${PIP} install -r prod.requirements.txt

lint:
	${FLAKE8} src/

run:
	${PYTHON} src/main.py
