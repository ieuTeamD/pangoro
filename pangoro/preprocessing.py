import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import seaborn as sns
import matplotlib.pyplot as plt
#import warnings


class PangoroDataFrame(pd.DataFrame):
    """
    The class is used to extend the properties of Dataframes to a preprocess its data. 
    It provides the end user with cleaning functions for each variable type:
        Numerical, Categorical Ordinal and Categorical Nominal    
    """

    #Initializing the inherited pd.DataFrame
    def __init__(self, *args, **kwargs):
        super(PangoroDataFrame,  self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
        
        '''        
        if len(args) == 1 and isinstance(args[0], PangoroDataFrame):
            args[0]._copy_attrs(self)
    def _copy_attrs(self, df):
        for attr in self._attributes_.split(","):
            df.__dict__[attr] = getattr(self, attr, None)
            '''

    @property
    def _constructor(self):
        '''
        Creates a self object that is basically a pandas.Dataframe.
        self is a dataframe-like object inherited from pandas.DataFrame
        self behaves like a dataframe + new custom attributes and methods.
        '''
        return PangoroDataFrame
    
    def echo(txt):
        '''
        takes a text input string and returns it as is 
        Parameters
        ----------
        txt: str
        
        Returns
        -------
        str
            text input is returned back to user
        '''
        return txt


    def plot_target_correlations(self, target_col='', save_as_png=False, color_map='BrBG'):
        '''
        Plot funtion to quickly show the correlations between the Target variable and the other variables in the PangoroDataFrame. Results can be save to png for permanent storage with specific name
        ----------
        target_col:  str, (Optional), Default = ''
            name of target column/variable
        save_as_png: bool, (Optional), Default = False
            specifies whether to save the output plot to png file or not
        color_map:  str, (Optional), Default = 'BrBG'
            color map based on Matplotlib defined color maps including sequential, diverging, cyclic, qualitative and misc. color maps (e.g. inferno and Purples) 
            Reference: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        Returns
        -------
        Correlation Heatmap Plot
            Correlation Plot of PangoroDataFrame Target Column/Feature with the other Columns/Features
        '''

        plt.figure(figsize=(8, 12))
        heatmap = sns.heatmap(self.corr()[[target_col]].sort_values(by=target_col, ascending=False), vmin=-1, vmax=1, annot=True, cmap=color_map)
        heatmap.set_title('Features Correlating with '+target_col, fontdict={'fontsize':18}, pad=16);
        # save heatmap as .png file
        # dpi - sets the resolution of the saved image in dots/inches
        # bbox_inches - when set to 'tight' - does not allow the labels to be cropped

        if(save_as_png==True):
            plt.savefig('pangoro_cor_heatmap_target_'+target_col+'.png', dpi=300, bbox_inches='tight')

        return
    
    def plot_all_correlations(self, save_as_png=False, color_map='BrBG'):
        
        '''
        Plot funtion to quickly show the correlation between all variablesin the PangoroDataFrame. Results can be save to png for permanent storage with specific name
        ----------
        save_as_png: bool, (Optional), Default = False
            specifies whether to save the output plot to png file or not
        color_map:  str, (Optional), Default = 'BrBG'
            color map based on Matplotlib defined color maps including sequential, diverging, cyclic, qualitative and misc. color maps (e.g. inferno and Purples) 
            Reference: https://matplotlib.org/stable/tutorials/colors/colormaps.html
        Returns
        -------
        Correlation Heatmap Plot
            Correlation Plot of all PangoroDataFrame Columns/Features 
        '''
        
        plt.figure(figsize=(16, 6))
    
        heatmap = sns.heatmap(self.corr(), vmin=-1, vmax=1, annot=True, cmap=color_map)
    
        heatmap.set_title('All Correlations Heatmap', fontdict={'fontsize':12}, pad=12);
        
        # save heatmap as .png file
        # dpi - sets the resolution of the saved image in dots/inches
        # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
    
        if(save_as_png==True):
            plt.savefig('pangoro_cor_heatmap_all'+'.png', dpi=300, bbox_inches='tight')
    
        return
    
    def numerical_transformation(self, col = [], na_treat = 'keep', na_fill = 0, knn_neighbors = 5, out_treat = False, 
                            out_upper = 0.75, out_lower = 0.25, scaling='no'):
        
        '''
        This function will automate cleaning of numerical features in Panda dataframe
        
        This fuction can perform the following 3 treatments:
        1. Treating null values using one of the following methods:
            a) Dropping null
            b) Replace them with mean
            c) Replace them with mode
            d) Replace them with min
            e) Replcae them with max
            f) Replace them with a specified number
            g) Impute them using KNN imputation algorithm
        2. Treating outliers by using the tukey method.
        3. Scaling the values using one the followings:
            a) MinMax Scaler
            b) Standard Scaler
        
        Parameters
        ----------
        col: list, (Optional), Default = [ ]:
            List of numerical columns name withing the PangoroDataFrame. If no value provided, the method will apply on all columns with numeric datatypes.
        
        na_treat: str, (Optional), Default = 'keep'
            Null values treatment method: 
                1. 'drop': to drop all nulls 
                2. 'mean': to replce nulls with the mean 
                3. 'mode': to replace nulls with the mode 
                4. 'min': to replace nulls with the minumum value 
                5. 'max': to replace nulls with maximum value 
                6. 'fill': to replace nulls with a specific number 
                7. 'knn_fill': to impute nulls using knn algorithm 
                8. 'keep': not to treat nulls.
        
        na_fill: float, (Optional), Default = 0
            If na_treat = 'fill', this parameter will be used to replace the null.
        
        knn_neighbors, int, (Optional), Default = 5
            If na_treat = 'knn_fill', this parameter will be used to indicate the number of neighbors for the knn method
        
        out_treat: bool, (Optional), Default = False
            This flag is to indicate if outliers treatment is needed or not.
        
        out_upper: float, (Optional), Default = 0.75
            This is to indicate the upper limit of the tukey method for treating outliers. Must be between 0 & 1.
        
        out_lower: float, (Optional), Default = 0.25
        This is to indicate the lower limit of the tukey method for treating outliers. Must be between 0 & 1.
        
        scaling: str, (Optional), Default = 'no'
            This is to indicate the type of scaling: 
                1. 'StandardScaler': to perform standard scaling 
                2. 'MinMaxScaler': to perofrm MinMax scaling 
                3. 'no': not to perform scaling
        Returns
        -------
        PangoroDataFrame
            PangoroDataFrame with cleaned and transformed numerical features
        '''
        
        # The following section will test the paramters
        if str(type(self)) != "<class 'pangoro.preprocessing.PangoroDataFrame'>":
            raise TypeError('Error: dataframe is not a valid PangoroDataFrame')
        if str(type(col)) != "<class 'list'>":
            raise TypeError('Error: paramter col must be a list')
        if str(type(na_treat)) != "<class 'str'>":
            raise TypeError('Error: paramter na_treat must be a string')
        if not(str(type(na_fill)) == "<class 'int'>" or str(type(na_fill)) == "<class 'float'>"):
            raise TypeError('Error: paramter na_fill must be a number')
        if str(type(knn_neighbors)) != "<class 'int'>":
            raise TypeError('Error: paramter knn_neighbors must be an integer')
        if str(type(out_treat)) != "<class 'bool'>":
            raise TypeError('Error: paramter out_treat must be a boolean')
        if not(str(type(out_upper)) == "<class 'int'>" or str(type(out_upper)) == "<class 'float'>") or out_upper <= 0 or out_upper > 1:
            raise TypeError('Error: paramter out_upper must be a number grater than 0 and less than or equal to 1')
        if not(str(type(out_lower)) == "<class 'int'>" or str(type(out_lower)) == "<class 'float'>") or out_lower < 0 or out_lower >= 1:
            raise TypeError('Error: paramter out_lower must be a number grater than or equal to 0 and less than 1')
        if str(type(scaling)) != "<class 'str'>":
            raise TypeError('Error: paramter scaling must be a string')
            
        # The following section will scan the data types of the columns looking for numerical columns.
        
        if col == []:
            for c in self.columns:
                if self[c].dtype.name == 'int64' or self[c].dtype.name == 'float64':
                    col.append(c)
        else:
            for c in col:
                if not(self[c].dtype.name == 'int64' or self[c].dtype.name == 'float64'):
                    raise TypeError('Error: column ' + c + ' is not numeric')
    
                
    #------------------------------------------------------------------
        
    # The following section will treat the na values of the columns
        
        if na_treat == 'drop':                    # If na_treat = 'drop', all NaN values for the specified columns will be droped
            self = self.dropna(subset = col, inplace=True)
            
        elif na_treat == 'mean':                  # If na_treat = 'mean', all NaN values for the specified columns will be replaced with the column average
            for c in col:
                mean = self[c].mean()
                self.loc[:, c].fillna(mean, inplace=True)
                
        elif na_treat == 'mode':                  # If na_treat = 'mode', all NaN values for the specified columns will be replaced with the column mode
            for c in col:
                modei = self[c].mode()[0]
                self.loc[:, c].fillna(modei, inplace=True)
                
        elif na_treat == 'min':                   # If na_treat = 'min', all NaN values for the specified columns will be replaced with the column minimum
            for c in col:
                mini = self[c].min()
                self.loc[:, c].fillna(mini, inplace=True)
                
        elif na_treat == 'max':                   # If na_treat = 'max', all NaN values for the specified columns will be replaced with the column maximum
            for c in col:
                maxi = self[c].max()
                self.loc[:, c].fillna(maxi, inplace=True)
                
        elif na_treat == 'fill':                  # If na_treat = 'fill', all NaN values for the specified columns will be replaced with the na_fill value (default is 0)
            for c in col:
                self.loc[:, c].fillna(na_fill, inplace=True)
                
        elif na_treat == 'knn_fill':              # If na_treat = 'knn_fill', all NaN values for the specified columns will be imputed using knn regressor
            imputer = KNNImputer(n_neighbors=knn_neighbors, copy = True)
            for c in col:
                self[c] = np.round_(imputer.fit_transform(self[[c]]))
    
            
        elif na_treat == 'keep':                  # If na_treat = 'keep', all NaN values for the specified columns will be kept with no change
            pass
            
        else:
            raise TypeError('Error: The na_treat variable ' + na_treat + ' is not a valid')
    
        #---------------------------------------------------------------------------------------------------    
        # The following section will treat outliers using tukey method
        
        if out_treat:
            if out_upper > out_lower and out_upper <= 1 and out_lower >= 0:
                upper_lower_dic = {}
                for c in col:
                    Q3 = self[c].quantile(out_upper)
                    Q1 = self[c].quantile(out_lower)
                    IQR = Q3 - Q1
                    lower_lim = Q1 - 1.5 * IQR
                    upper_lim = Q3 + 1.5 * IQR
                    upper_lower_dic[c+'_L'] = lower_lim
                    upper_lower_dic[c+'_U'] = upper_lim
                for c in col:
                    self.drop(self[self[c] < upper_lower_dic[c+'_L']].index, inplace=True)
                    self.drop(self[self[c] > upper_lower_dic[c+'_U']].index, inplace=True)
                    
            else:
                raise TypeError('Error: parameters out_upper and/or out_lower are not valid')
        #---------------------------------------------------------------------------------------------------
        
        # The following section will perform scaling
        if scaling == 'StandardScaler':
            scaler = StandardScaler()
            self[col] = scaler.fit_transform(self[col])
        elif scaling == 'MinMaxScaler':
            scaler = MinMaxScaler()
            self[col] = scaler.fit_transform(self[col])
        elif scaling == 'no':
            pass
        else:
            raise TypeError('Error: scaler ' + scaling + ' is not valid')
        #---------------------------------------------------------------------------------------------------
        
        # Return the final dataframe
        return self
    
    
    #*********************************************
    def categorical_nominal_transformation(self, col = [], na_treat = 'keep', na_fill = 'missing', knn_neighbors = 5, 
                                         transform = False, maximum_cat = 10, start_from = 0, increment_by = 1):
        
        '''
        This function will automate cleaning of catagorical nominal features in Panda dataset
        
        This fuction can perform the following 2 treatments:
        1. Treating null values using one of the following methods:
            a) Dropping null
            b) Replace them with mode
            c) Replace them with a specified string
            d) Impute them using KNN imputation algorithm
        2. Transform the catagorical features into number by:
            a) Using one hot encoding 
            b) Replacing the each category by a specific value:
        
        Parameters
        ----------
        
        col: list, (Optional), Default = [ ]:
            List of categorical columns name withing the PangoroDataFrame. If no value provided, the method will apply on all columns with categorical or object datatypes.
        
        na_treat: str, (Optional), Default = 'keep'
            Null values treatment method: 1. 'drop': to drop all nulls 2. 'mode': to replace nulls with the mode 3. 'fill': to replace nulls with a specific text 4. 'knn_fill': to impute nulls using knn algorithm 5. 'keep': not to treat nulls.
        
        na_fill: float, (Optional), Default = 'missing'
            If na_treat = 'fill', this parameter will be used to replace the null.
        
        knn_neighbors, int, (Optional), Default = 5
            If na_treat = 'knn_fill', this parameter will be used to indicate the number of neighbors for the knn method
        
        transform: bool, (Optional), Default = False
            This flag is to indicate if transformation is needed or not.
        
        maximum_cat: int, float, (Optional), Default = 10
            This indecate the limit of the number of categories in each column where the method will perform one hot encoding or replacing the categories with sequence numbers starting from 0.
        
        start_from: int, float, (Optional), Default = 0
            This indicates the start value of the sequence when replacing the categories with a sequence.
        
        Returns
        -------
        PangoroDataFrame
            PangoroDataFrame with cleaned and transformed categorical nominal features
        '''
        # The following section will test the paramters
        if str(type(self)) != "<class 'pangoro.preprocessing.PangoroDataFrame'>":
            raise TypeError('Error: dataframe is not a valid PangoroDataFrame')
        if str(type(col)) != "<class 'list'>":
            raise TypeError('Error: paramter col must be a list')
        if str(type(na_treat)) != "<class 'str'>":
            raise TypeError('Error: paramter na_treat must be a string')
        if str(type(na_fill)) != "<class 'str'>":
            raise TypeError('Error: paramter na_fill must be a string')
        if str(type(knn_neighbors)) != "<class 'int'>":
            raise TypeError('Error: paramter knn_neighbors must be an integer')
        if str(type(transform)) != "<class 'bool'>":
            raise TypeError('Error: paramter transform must be a boolean')
        if str(type(maximum_cat)) != "<class 'int'>" or maximum_cat < 0:
            raise TypeError('Error: paramter maximum_cat must be an integer number greater than or equal to 0')
        if not(str(type(start_from)) == "<class 'int'>" or str(type(start_from)) == "<class 'float'>"):
            raise TypeError('Error: paramter start_from must be a number')
        if not(str(type(increment_by)) == "<class 'int'>" or str(type(increment_by)) == "<class 'float'>"):
            raise TypeError('Error: paramter increment_by must be a number')
            
        # The following section will scan the data types of the columns looking for catagorical columns.
        if col == []:
            for c in self.columns:
                if self[c].dtype.name == 'object' or self[c].dtype.name == 'category':
                    col.append(c)
                    
    
        #---------------------------------------------------------------------------------------------------
        
        # The following section will treat the na values of the columns
        knn = False
        if na_treat == 'drop':                    # If na_treat = 'drop', all NaN values for the specified columns will be droped
            self = self.dropna(subset = col, inplace=True)
                
        elif na_treat == 'mode':                  # If na_treat = 'mode', all NaN values for the specified columns will be replaced with the column mode
            for c in col:
                modei = self[c].mode()[0]
                self.loc[:, c].fillna(modei, inplace=True)
                
        elif na_treat == 'fill':                  # If na_treat = 'fill', all NaN values for the specified columns will be replaced with the na_fill value (default is 'missing')
            for c in col:
                self.loc[:, c].fillna(na_fill, inplace=True)
                
        elif na_treat == 'knn_fill':              # If na_treat = 'knn_fill', all NaN values for the specified columns will be imputed using knn regressor This step will be done after transformation
            knn = True
            
        elif na_treat == 'keep':                  # If na_treat = 'keep', all NaN values for the specified columns will be kept with no change
            pass
            
        else:
            raise TypeError('Error: The na_treat variable ' + na_treat + ' is not a valid')
        #---------------------------------------------------------------------------------------------------
        
        if transform:
            dropped_col = []
            for c in col:
                if len(self[c].unique()) <= maximum_cat: 
                    df_in = self
                    ohe = OneHotEncoder(sparse=False, drop='first')
                    ohe.fit(df_in[[c]])
                    self = pd.DataFrame(ohe.transform(df_in[[c]]),
                    columns = ohe.get_feature_names_out([c]))
                    self.set_index(df_in.index, inplace=True)
                    self = pd.concat([df_in, self], axis=1).drop([c], axis=1)
                    dropped_col.append(c)
                else:
                    for i in range(start_from,(len(self[c].unique())*increment_by) + start_from, increment_by):
                        self.loc[self[c] == self[c].unique()[int((i-start_from)/increment_by)], c] = i
                col = list(set(col)-set(dropped_col))
        
        if knn: # If na_treat = 'knn_fill', all NaN values for the specified columns will be imputed using knn
            imputer = KNNImputer(n_neighbors=knn_neighbors, copy = True)
            for c in col:
                self[c] = np.round_(imputer.fit_transform(self[[c]]))
    

        #---------------------------------------------------------------------------------------------------
        # Return the final dataframe
        return self
    #*********************************************
    def categorical_ordinal_transformation(self, col = [], na_treat = 'keep', na_fill = 'missing', knn_neighbors = 5, 
                                         transform = False, cat_dict = 'auto', start_from = 0, increment_by = 1):
        
        '''
        
        This function will automate cleaning of catagorical ordinal features in Panda dataset
        
        This fuction can perform the following 2 treatments:
        1. Treating null values using one of the following methods:
            a) Dropping null
            b) Replace them with mode
            c) Replace them with a specified string
            d) Impute them using KNN imputation algorithm
        2. Transform the catagorical features into number by replacing them with a squence number
        
        Parameters
        ----------
        
        col: list, (Optional), Default = [ ]:
            List of categorical columns name withing the PangoroDataFrame. If no value provided, the method will apply on all columns with categorical or object datatypes.
        
        na_treat: str, (Optional), Default = 'keep'
            Null values treatment method: 1. 'drop': to drop all nulls 2. 'mode': to replace nulls with the mode 3. 'fill': to replace nulls with a specific text 4. 'knn_fill': to impute nulls using knn algorithm 5. 'keep': not to treat nulls.
        
        na_fill: float, (Optional), Default = 'missing'
            If na_treat = 'fill', this parameter will be used to replace the null.
        
        knn_neighbors, int, (Optional), Default = 5
            If na_treat = 'knn_fill', this parameter will be used to indicate the number of neighbors for the knn method
        
        transform: bool, (Optional), Default = False
        This flag is to indicate if transformation is needed or not.
        
        cat_dict: dict, str, (Optional), Default = 'auto'
            If cat_dict = 'auto', the algorithm will sort the each column category alphabetically and assign values from start_from parameters incrementing by increment_by parameters. Otherwise, when dictionary is supplied, the algorithm will replace the current value with the new one according to the dictionary. Example: {'poor':0, 'fair':1, 'good':2, 'v.good':3, 'excellent':4}
        
        start_from: int, float, (Optional), Default = 0
            This indicates the start value of the sequence when replacing the categories with a sequence when cat_dict = 'auto'
        
        increment_by: int, float, (Optional), Default = 1
            This value is how much the sequence is incremented by when cat_dict = 'auto'
        
        Returns
        -------
        PangoroDataFrame
            PangorDataframe with cleaned and transformed categorical ordinal features
                
        '''
        
        # The following section will test the paramters
        if str(type(self)) != "<class 'pangoro.preprocessing.PangoroDataFrame'>":
            raise TypeError('Error: dataframe is not a valid PangoroDataFrame')
        if str(type(col)) != "<class 'list'>":
            raise TypeError('Error: paramter col must be a list')
        if str(type(na_treat)) != "<class 'str'>":
            raise TypeError('Error: paramter na_treat must be a string')
        if str(type(na_fill)) != "<class 'str'>":
            raise TypeError('Error: paramter na_fill must be a string')
        if str(type(knn_neighbors)) != "<class 'int'>":
            raise TypeError('Error: paramter knn_neighbors must be an integer')
        if str(type(transform)) != "<class 'bool'>":
            raise TypeError('Error: paramter transform must be a boolean')
        if not(str(type(cat_dict)) == "<class 'dict'>" or str(type(cat_dict)) == "<class 'str'>"):
            raise TypeError("Error: paramter cat_dict must be a dictionary or 'auto'")
        if not(str(type(start_from)) == "<class 'int'>" or str(type(start_from)) == "<class 'float'>"):
            raise TypeError('Error: paramter start_from must be a number')
        if not(str(type(increment_by)) == "<class 'int'>" or str(type(increment_by)) == "<class 'float'>"):
            raise TypeError('Error: paramter increment_by must be a number')
        
        # The following section will scan the data types of the columns looking for catagorical columns.
        if col == []:
            for c in self.columns:
                if self[c].dtype.name == 'object' or self[c].dtype.name == 'category':
                    col.append(c)
    
        #---------------------------------------------------------------------------------------------------
        
        # The following section will treat the na values of the columns
        knn = False
        if na_treat == 'drop':                    # If na_treat = 'drop', all NaN values for the specified columns will be droped
            self = self.dropna(subset = col, inplace=True)
                
        elif na_treat == 'mode':                  # If na_treat = 'mode', all NaN values for the specified columns will be replaced with the column mode
            for c in col:
                modei = self[c].mode()[0]
                self.loc[:, c].fillna(modei, inplace=True)
                
        elif na_treat == 'fill':                  # If na_treat = 'fill', all NaN values for the specified columns will be replaced with the na_fill value (default is 'missing')
            for c in col:
                self.loc[:, c].fillna(na_fill, inplace=True)
                
        elif na_treat == 'knn_fill':              # If na_treat = 'knn_fill', all NaN values for the specified columns will be imputed using knn regressor This step will be done after transformation
            knn = True
            
        elif na_treat == 'keep':                  # If na_treat = 'keep', all NaN values for the specified columns will be kept with no change
            pass
            
        else:
            raise TypeError('Error: The na_treat variable ' + na_treat + ' is not a valid')
        #---------------------------------------------------------------------------------------------------
        
        if transform:
            if cat_dict == 'auto':
                for c in col:
                    c_list = sorted(x for x in self[c].unique() if pd.isnull(x) == False)   
                    for i in range(start_from,(len(c_list) * increment_by) + start_from, increment_by):
                        self.loc[self[c] == c_list[int((i-start_from)/increment_by)], c] = i
            else:
                for c in col:
                    self[c] = self[c].replace(cat_dict)
                        
        if knn:                                   # If na_treat = 'knn_fill', all NaN values for the specified columns will be imputed using knn regressor
            imputer = KNNImputer(n_neighbors=knn_neighbors, copy = True)
            for c in col:
                self[c] = np.round_(imputer.fit_transform(self[[c]]))
    
        #---------------------------------------------------------------------------------------------------
        
        # Return the final dataframe
        return self
    
    
    def split_transform(self, col = [], train_size = 0.75, random_state = None, shuffle = True, na_treat = 'keep', 
                    na_fill = 'missing', transform_type = 'mean', target_col = None):
    
    # The following section will test the paramters
        '''
        This function will split and transform PangoroDataframe by using the target variable
        This fuction can perform the following 3 treatments:
        1. Treating null values using one of the following methods:
            a) Dropping null
            b) Replace them with mode
            c) Replace them with a specified string
        2. Split the data into train and test sets
        3. Transform the catagorical features into number using the traget variable from the training set.
        
        Parameters
        ----------
        
        col: list, (Optional), Default = [ ]:
            List of categorical columns name withing the PangoroDataFrame. If no value provided, the method will apply on all columns with categorical or object datatypes.
        
        train_size: float, (Optional), Default = 0.75:
            This parameter represents the proportion of the dataset to include in the train split and should be between 0.0 and 1.0. The test size will be automatically set to 1 - train_size. If int, represents the absolute number of train samples in this case the test size will be automatically set to number of records - train_size.
        
        random_state: int (Optional), Default = None:
            Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
        
        shuffle: boolean (Optional), Default = True:
            Whether or not to shuffle the data before splitting. If shuffle=False then stratify must be None.
        
        na_treat: str, (Optional), Default = 'keep'
            Null values treatment method: 1. 'drop': to drop all nulls 2. 'mode': to replace nulls with the mode 3. 'fill': to replace nulls with a specific text 4. 'keep': not to treat nulls.
        
        na_fill: float, (Optional), Default = 'missing'
            If na_treat = 'fill', this parameter will be used to replace the null.
        
        transform_type: str, (Optional), Default = 'mean'
            This to indicate the transformation type. The method will transform the catigorical variable based on the target variable for each category. 1. 'mean': transform using target's mean 2. 'min': transform using target's min 3. 'max': transform using target's max
        
        target_col: str, (Optional), Default = 'None'
            This parameter is the target variable that will be used for transformation. When target_col = None, there will be no transformation.
        
        Returns
        -------
        PangoroDataFrame, PangoroDataFrame
            Two PangoroDataframes with cleaned and transformed categorical ordinal features:
                PangoroDataFrame(df_train), PangoroDataFrame(df_test)
                                                                     
        '''
        if str(type(self)) != "<class 'pangoro.preprocessing.PangoroDataFrame'>":
            raise TypeError('Error: df is not a valid PangoroDataFrame')
        if str(type(col)) != "<class 'list'>":
            raise TypeError('Error: paramter col must be a list')
        if not(str(type(train_size)) == "<class 'float'>" or str(type(train_size)) == "<class 'int'>"):
            raise TypeError('Error: paramter train_size must be a number between 0 & 1')
        if not(str(type(random_state)) == "<class 'int'>" or random_state == None):
            raise TypeError('Error: paramter random_state must be an integer number')
        if str(type(shuffle)) != "<class 'bool'>":
            raise TypeError('Error: paramter shuffle must be a boolean')
        if str(type(na_treat)) != "<class 'str'>":
            raise TypeError('Error: paramter na_treat must be a string')
        if str(type(na_fill)) != "<class 'str'>":
            raise TypeError('Error: paramter na_fill must be a string')
        if str(type(transform_type)) != "<class 'str'>":
            raise TypeError('Error: paramter transform_type must be a string')
        if not (str(type(target_col)) != "<class 'str'>" or target_col != None):
            raise TypeError('Error: paramter target_col must be a string')
    
        
        # The following section will scan the data types of the columns looking for catagorical columns.
        if col == []:
            for c in self.columns:
                if self[c].dtype.name == 'object' or self[c].dtype.name == 'category':
                    col.append(c)
    
        #---------------------------------------------------------------------------------------------------
        
        # The following section will treat the na values of the columns
        
        if na_treat == 'drop':                    # If na_treat = 'drop', all NaN values for the specified columns will be droped
            self = self.dropna(subset = col)
                
        elif na_treat == 'mode':                  # If na_treat = 'mode', all NaN values for the specified columns will be replaced with the column mode
            for c in col:
                mode = self[c].mode()
                self.loc[:, c].fillna(mode[0], inplace=True)
                
        elif na_treat == 'fill':                  # If na_treat = 'fill', all NaN values for the specified columns will be replaced with the na_fill value (default is 'missing')
            for c in col:
                self.loc[:, c].fillna(na_fill, inplace=True)
                
        elif na_treat == 'keep':                  # If na_treat = 'keep', all NaN values for the specified columns will be kept with no change
            pass
            
        else:
            raise TypeError('Error: The na_treat parameter ' + na_treat + ' is not a valid')
        #---------------------------------------------------------------------------------------------------
        
        # The folllowing section will split the data into train and test
        df_train, df_test = train_test_split(self, train_size = train_size, random_state = random_state, 
                                             shuffle = shuffle)
        
        #---------------------------------------------------------------------------------------------------
        for c in col:           
            if target_col != None:
                if transform_type == 'mean':
                    df_train_temp = df_train.groupby(c)[
                        [target_col]].mean().sort_values(by=[target_col], ascending=False).rename(
                        columns={target_col: 'result_' + target_col})
                            
                elif transform_type == 'max':
                    df_train_temp = df_train.groupby(c)[
                        [target_col]].max().sort_values(by=[target_col], ascending=False).rename(
                        columns={target_col: 'result_' + target_col})
    
                elif transform_type == 'min':
                    df_train_temp = df_train.groupby(c)[
                        [target_col]].min().rename(columns={target_col: 'result_' + target_col})
                
                else:
                    raise TypeError('Error: The transform_type parameter ' + transform_type + ' is not a valid')
                
                df_train_temp_dict = df_train_temp.to_dict()['result_' + target_col]
                df_train[c] = df_train[c].replace(df_train_temp_dict)
                df_test[c] = df_test[c].replace(df_train_temp_dict)
    
        #---------------------------------------------------------------------------------------------------
        
        # Return the final dataframe
        return PangoroDataFrame(df_train), PangoroDataFrame(df_test)
