## Description
## To setup
Create an environment using
```bash
pip install virtualenv 
```
Mac:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

To install the packages in the venv environment run:

```bash
pip install --upgrade setuptools wheel
pip install -r requirements.txt
```

## To run
The run_model.py file contains the necessary code to run an mlflow experiment.
The compare_metrics.py file contains code necessary for getting metrics out.
