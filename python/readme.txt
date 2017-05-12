python KDD cup 99 classification project

REQUIRED FOLDER STRUCTURE

root
|-doc
   |-P15240916-IMAT5235-ANN-Assignment-2.docx
   |-data.xlsx
|-python
   |-MLP_*.py (classifier versions)
   |-kddcup.py
   |-CART.py
   |-KNN.py
   |-LOGRES.py
   |-MLP.py
   |-NBAYES.py
   |-SVM.py
|-data
   |-kddcup.data_10_percent_corrected
|-analysis

Use anaconda to configure your python environment dependencies

Windows:
>conda create --name KDDCUP99 python=2
>source activate KDDCUP99

OS X and Linux:
>conda create --name KDDCUP99 python=2
>activate KDDCUP99

OS X and Linux:
>conda create –n KDDCUP99
>source activate KDDCUP99

Install dependencies.
>pip install scikit-learn
>pip install pandas
>pip install numpy
>pip install scipy

Note!
On Windows, you may/will have issues installing scipy from packages
So, download the appropriate python *.whl file from here:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy
Then install from where you downloaded the whl file locally

Note!
If you get (yet another) issue when running the python scripts
"ImportError: cannot import name NUMPY_MKL"
Then you will also need to install the numpy+mkl whl file from:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#snumpy

Alternatively, if you are on Windows and can't get the python dependencies to install...
download VirtualBox and a recent Ubuntu distribution and use a virtual machine.

To check which packages have been installed:
>pip list

numpy (1.11.3+mkl)
pandas (0.20.0)
pip (9.0.1)
python-dateutil (2.6.0)
pytz (2017.2)
scikit-learn (0.18.1)
scipy (0.19.0)
setuptools (27.2.0)
six (1.10.0)
wheel (0.29.0)

EXECUTING THE CLASSIFIERS 
python MLP*.py

