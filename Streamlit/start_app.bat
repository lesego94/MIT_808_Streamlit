@echo off

REM check if the venv directory exists
if not exist "venv\" (
    REM if it does not exist, create a new virtual environment
    python -m venv venv
)

REM activate the virtual environment
call venv\Scripts\activate

REM install requirements
pip install -r src\requirements.txt


REM run the Streamlit app
streamlit run src\main.py
