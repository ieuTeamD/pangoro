import unittest
from pangoro.preprocessing import PangoroDataFrame
import pandas as pd


def broken_function():
    raise Exception('This is broken')

class TestDataFrame(unittest.TestCase):
    def test_simple_dataframe(self):
        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        self.assertEqual(True, df1.equals(df2))

    def test_numerical_transformation_unchanged(self):   
        sample_data = pd.DataFrame({'pet':      ['cat', 'dog', 'dog', pd.NA, 'cat', 'dog', 'cat', 'fish'],
                           'color': ['white', 'brown', 'black', 'gold', pd.NA,'black', 'white', 'silver'],
                           'weight':   [50.5, 22.5, 4, pd.NA , 12, 39, 16, pd.NA]})
        sample_data['weight'] = pd.to_numeric(sample_data['weight'])
        #sample_data.dtypes
    
        pdf = PangoroDataFrame(sample_data)
        pdf.numerical_transformation()# without passing any parameters, this should not perform any change
        self.assertEqual(True, pdf.equals(sample_data))

    def test_numerical_transformation_na_treat_mode(self):
        df1 = pd.DataFrame({'a': [1, pd.NA,1,1,1,1,1,8,9.0,pd.NA]})
        df2 = pd.DataFrame({'a': [1, 1,1,1,1,1,1,8,9,1.0]})
        df1['a'] = pd.to_numeric(df1['a'])
        df2['a'] = pd.to_numeric(df2['a'])

        pdf = PangoroDataFrame(df1)
        pdf.numerical_transformation(col=['a'],na_treat = 'mode')# treating NAs with mode should yield same result as 
        self.assertEqual(True, pdf.equals(df2))


    def test_numerical_transformation_raise_TypeError(self):
        df1_with_err = pd.DataFrame({'a': [1,'cat']})
        pdf = PangoroDataFrame(df1_with_err)

        with self.assertRaises(TypeError):
            pdf.numerical_transformation(col=['a'],na_treat = 'mode')# TypeError should be raised            
        self.assertTrue('TypeError: Error: column a is not numeric' )
        
    def test_categorical_transformation_Transform(self):
        df1 = pd.DataFrame({'a': ['cat', pd.NA,'dog',pd.NA]})
        df2 = pd.DataFrame({'a_dog': [0.0,pd.NA,1.0]})
        df2=df2.dropna()
        df2['a_dog']=pd.to_numeric(df2['a_dog'],errors='coerce')
         
        pdf = PangoroDataFrame(df1)        
        
        pdf.categorical_nominal_transformation(col=['a'],na_treat = 'drop' )
        pdf =pdf.categorical_nominal_transformation(col=['a'],transform=True )
        pdf.equals(df2)
        self.assertEqual(True, pdf.equals(df2))

if __name__ == '__main__':
    unittest.main()
