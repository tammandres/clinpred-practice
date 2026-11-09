# Machine learning for clinical prediction

This code repository accompanies the lecture "Machine learning for clinical prediction" I gave for the MSc in Applied Digital Health programme at the University of Oxford (5 November 2026).

It includes a brief example of how penalised logistic regression (PLR), explainable boosting machine (EBM), and xgboost (XGB) can be applied for predicting a binary outcome (such as a cancer diagnosis).


## Usage

There are at least three ways to use this repository.

1. First, you could open the `example.ipynb` jupyter notebook on Binder: 

   https://mybinder.org/v2/gh/tammandres/clinpred-practice/HEAD?labpath=example.ipynb

   If you choose that route, it may take some time for the environment to install but that should happen automatically.

2. Second, you could open the notebook in Google Colab:

   https://colab.research.google.com/github/tammandres/clinpred-practice/blob/main/example.ipynb

   This requires a Google Account. 
   
   On Colab, you also need to uncomment the lines in the first code cell, to install packages, so the original code block
    ```bash
    #%pip install interpret==0.7.8
    #%pip install optuna==5.0.0
    ```

    becomes
    ```bash
    %pip install interpret==0.7.8
    %pip install optuna==5.0.0
    ```

3. Finally, to run this code on your local computer:

    * Install miniconda, https://www.anaconda.com/docs/getting-started/miniconda/install/windows-gui-install

    * Install Visual Studio Code (VSCode), https://code.visualstudio.com/

    * Inside VSCode, install python and Jupyter extensions

    * Then open VSCode -> Terminal -> New Terminal, and run the following
        ```
        conda create -n clinpred python=3.9
        pip install -r requirements.txt
        conda activate clinpred
        ```

        This will create a virtual environment that contains python and required python packages.
        Virtual environments are good for creating an isolated setup that has specific versions of the packages.

    * When using VSCode, set "clinpred" as the default conda environment: "cmd + shift + P" -> "Python: Select Interpreter" -> clinpred.

    * Finally, click on the `example.ipynb` file and it should be usable inside VSCode.

    I have not been able to fully test these instructions, apologies!
    

You could e-mail me if anything is not working: andres.[surname]@wolfson.ox.ac.uk


## Files

* `example.py` : a plain python script that illustrates how each type of model can be applied to data

* `example.ipynb` : the same as example.py, but in an interactive notebook format


