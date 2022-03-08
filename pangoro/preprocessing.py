import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
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

    @property
    def _constructor(self):
        return PangoroDataFrame
        
    
    def echo(txt):
        '''
        takes a text string and returns it as is 
        Parameters
        ----------
        txt: str
        
        Returns
        -------
        str
        text input is returned back to user
        '''
        return txt


    def plot_target_correlation(self, target_col='', save_as_png=False, color_map='BrBG'):
        plt.figure(figsize=(8, 12))
        heatmap = sns.heatmap(self.corr()[[target_col]].sort_values(by=target_col, ascending=False), vmin=-1, vmax=1, annot=True, cmap=color_map)
        heatmap.set_title('Features Correlating with '+target_col, fontdict={'fontsize':18}, pad=16);
    
        return
    
    def plot_all_correlation(self, save_as_png=False, color_map='BrBG'):
        # Increase the size of the heatmap.
        plt.figure(figsize=(16, 6))# TODO: Make dynamically change per df col num
        # Store heatmap object in a variable to easily access it when you want to include more features (such as title).
        # Set the range of values to be displayed on the colormap from -1 to 1, and set the annotation to True to display the correlation values on the heatmap.
    
    
        heatmap = sns.heatmap(self.corr(), vmin=-1, vmax=1, annot=True, cmap=color_map)
    
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
        
        # save heatmap as .png file
        # dpi - sets the resolution of the saved image in dots/inches
        # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
    
        if(save_as_png==True):
            plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
    
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
            List of numerical columns name withing the supplied datafram. If no value provided, the method will apply on all columns with numeric datatypes.
        
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
            DataFrame
                Dataframe with cleaned numerical features
        '''
        
        # The following section will test the paramters
        if str(type(self)) != "<class 'pangoro.preprocessing.PangoroDataFrame'>":
            raise TypeError('Error: df is not a valid pandas datafram')
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
            self = self.dropna(subset = col)
            
        elif na_treat == 'mean':                  # If na_treat = 'mean', all NaN values for the specified columns will be replaced with the column average
            for c in col:
                mean = self[c].mean()
                self.loc[:, c].fillna(mean, inplace=True)
                
        elif na_treat == 'mode':                  # If na_treat = 'mode', all NaN values for the specified columns will be replaced with the column mode
            for c in col:
                mode = self[c].mode()
                self.loc[:, c].fillna(mode, inplace=True)
                
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
            List of categorical columns name withing the supplied datafram. If no value provided, the method will apply on all columns with categorical or object datatypes.
        
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
            DataFrame
                Dataframe with cleaned categorical nominal features
        '''
        # The following section will test the paramters
        if str(type(self)) != "<class 'pangoro.preprocessing.PangoroDataFrame'>":
            raise TypeError('Error: df is not a valid pandas datafram')
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
            self = self.dropna(subset = col)
                
        elif na_treat == 'mode':                  # If na_treat = 'mode', all NaN values for the specified columns will be replaced with the column mode
            for c in col:
                mode = self[c].mode()
                self.loc[:, c].fillna(mode[0], inplace=True)
                
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
                    columns = ohe.get_feature_names([c]))
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
            List of categorical columns name withing the supplied datafram. If no value provided, the method will apply on all columns with categorical or object datatypes.
        
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
            DataFrame
                Dataframe with cleaned categorical ordinal features
                
        '''
        
        # The following section will test the paramters
        if str(type(self)) != "<class 'pangoro.preprocessing.PangoroDataFrame'>":
            raise TypeError('Error: df is not a valid pandas datafram')
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
            self = self.dropna(subset = col)
                
        elif na_treat == 'mode':                  # If na_treat = 'mode', all NaN values for the specified columns will be replaced with the column mode
            for c in col:
                mode = self[c].mode()
                self.loc[:, c].fillna(mode[0], inplace=True)
                
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
