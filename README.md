### Project description 

![](https://svgshare.com/i/euv.svg)


### Pangoro: powerful Python data cleaning toolkit
## What is it?
**Pangoro** is a Python package that provides fast and flexible methods for cleaning numerical and categorical features in a dataframe. It aims to be a fundamental tool for doing data wrangling in Python.

### Main Features:
* For numerical features, pangoro provides the following tools:
  * Handle NA, drop, replace with mean, replace with mode, lamda function, replace with a number, replace with min replace with max, Use KNN classifications. Keep
  * Handle outliers, Keep, Percentile
  * Scaling, standard or min max
  * Convert to nuemerical
  * Scan and apply to all numerical or supply a list of features
* For categorical nominal features, pangoro provides the following tools:
  * Handle NA, Drop, replace with mode, use KNN classification for imputation.
  * Replace with sequence numbers based on supplied dictionary or based on alphabetical order.
  * Scan and apply to all Categorical or supply a list of features
  
### Where to get it:
The source code is currently hosted on [GitHub](https://github.com/ieuTeamD/pangoro)<br />
Binary installers for the latest released version are available at the Python Package Index [PyPi](https://pypi.org/project/pangoro/)<br />

~~~
pip install --upgrade pangoro
~~~
### Usage

#### 1. Import

Import what you need from the pangoro package. The choices are:
 ``PangoroDataFrame``, a class for cleaning numerical and categorical features in a dataframe in addition to plotting features correlations.

For this demonstration, we will import:

    >>> from pangoro.preprocessing import PangoroDataFrame

For these examples, we'll also use pandas to :

    >>> import pandas as pd

#### 2. Load some Data

Typicall data will is read from a tabular formatted file, but for illustration we'll create a simple dataframe from a Python dict:

    >>> sample_data = pd.DataFrame({'pet':      ['cat', 'dog', 'dog', pd.NA, 'cat', 'dog', 'cat', 'fish'],
                           'color': ['white', 'brown', 'black', 'gold', pd.NA,'black', 'white', 'silver'],
                           'weight':   [50.5, 22.5, 4, pd.NA , 12, 39, 16, pd.NA]})
    >>> sample_data['weight'] = pd.to_numeric(sample_data['weight'])
    >>> sample_data.dtypes

                           
#### 3. Create Instance and Apply Transformation

3.1 We will start by creating an instance of PangoroDataFrame using the created sample_data DataFrame above:

    >>> pdf = PangoroDataFrame(sample_data)

3.2 Then, we will automate cleaning of numerical features in PangoroDataFrame by replacing NAs with mean and without performing any scaling:

    >>> pdf.numerical_transformation(col = ['weight'], na_treat = 'mean', out_treat = True, out_upper = 0.8,
                             out_lower = 0.2, scaling='no', knn_neighbors = 0)

Similarly, as we continue with this example and using **pdf** as an instance **PangoroDataFrame** we can apply the other functions to clean and preprocess categorical ordinal by calling **_pdf.categorical_ordinal_transformation_** or categorical nominal by calling **_pdf.categorical_ordinal_transformation_** 

In addition, we can quickly plot correlations between features by calling: **_pdf.plot_all_correlations_**

---
**NOTE**
PangoroDataFrame object has to be re-assigned to itself in order to store changes inplace after transforming nominal categories, which will result in adding new features to the original PangoroDataFrame, for example:

    >>> pdf =pdf.categorical_nominal_transformation(col=['weight'],transform=True )

---

#### 4. Getting Help

To learn more about we can issue the following command:

    >>> help(PangoroDataFrame)

As a result we get detailed information about the package and each function included.

#### 5. Unit Test
pangoro package is equipped with test module that contains TestDataFrame class to perform all unit tests required to validate output of the PangoroDataFrame functions. Tests can be called by the follwoing command from the main pangoro package folder:

    >>> python -m tests.test

As a result, all test functions will be called and performed and the output will be the number of performed tests along with results of the tests (i.e. OK or FAILED(errors))

### License
[MIT](https://pypi.org/project/pangoro/)

### Background
Work on pangoro started in 2022 by a group of IE University students and has been under active development since then.

### Contributing to pangoro  
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.
Please contact us on [GitHub](https://github.com/ieuTeamD/pangoro)<br />

