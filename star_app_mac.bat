#!/bin/bash

# check if the venv directory exists
if [ ! -d "venv" ]
then
    # if it does not exist, create a new virtual environment
    python3 -m venv venv
fi

# create a new virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install requirements
pip install -r src/requirements.txt

# run the Streamlit app
streamlit run src/main.py
