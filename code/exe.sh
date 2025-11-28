#!/bin/bash

python -m venv .venv
source .venv/bin/activate
python -m pip install requirements.txt
python -m code.main
tar -cvzf results.tgz results/
rm -rf results/*