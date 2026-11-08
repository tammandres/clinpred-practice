# Machine learning for clinical prediction

This code repository accompanies the lecture "Machine learning for clinical prediction" I gave for the MSc in Applied Digital Health programme at the University of Oxford (5 November 2026).

It includes a brief example of how penalised logistic regression, Explainable Boosting Machine, and XGBoost can be applied for predicting a binary outcome (such as cancer diagnosis).


## Usage

To run this code on your local computer:

1. Install miniconda, https://www.anaconda.com/docs/getting-started/miniconda/install/windows-gui-install

2. Install Visual Studio Code (VSCode), https://code.visualstudio.com/

3. Then open VSCode -> Terminal -> New Terminal, and run the following
```
conda create -n clinpred python=3.9
pip install -r requirements.txt
conda activate clinpred
```

This will create a virtual environment that contains python and required python packages.
Virtual environments are good for creating an isolated setup that has specific versions of the packages.

When using visual studio code, set "clinpred" as the default conda environemnt "cmd + shift + P" -> "Python: Select Interpreter" -> clinpred.


## Files

* `example.py` : a plain python script that illustrates how each type of model can be applied to data

* `example.ipynb` : the same as example.py, but in an interactive notebook format


