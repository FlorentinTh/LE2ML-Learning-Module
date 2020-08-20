PYTHON=.venv/Scripts/python.exe
PIP=.venv/Scripts/pip.exe
FLAKE8=.venv/Scripts/flake8.exe

init:
	python -m venv .venv
	.venv\scripts\activate

install:
	${PYTHON} -m pip install --upgrade pip
	${PIP} install -r dev.requirements.txt

lint:
	${FLAKE8} src/

run:
	${PYTHON} src/main.py
