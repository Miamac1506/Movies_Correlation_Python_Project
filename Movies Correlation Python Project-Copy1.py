#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Setting up the environment for data visualization using the Pandas, Seaborn, and Matplotlib libraries

import pandas as pd #Importing the Pandas library, typically used for data manipulation and analysis.
import seaborn as sns #Importing the Seaborn library, which is used for statistical data visualization. 
import numpy as np

import matplotlib #Importing the Matplotlib library, which is a powerful library for creating static, animated, and interactive visualizations in Python.
import matplotlib.pyplot as plt # Importing the pyplot module from Matplotlib 
plt.style.use('ggplot') #Plot style
from matplotlib.pyplot import figure #The figure function is typically used to create a new figure or adjust the properties of the current figure. 

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) # Adjust the configuration of the plots we will create

df = pd.read_csv(r'/Users/macminhanh/Downloads/movies.csv') 


# In[5]:


#Look at the data

df.head()


# In[6]:


# Cleaning up the data

# Fidning missing data

for col in df.columns: #loop that iterates through each column in the DataFrame 
    pct_missing = np.mean(df[col].isnull()) #Within the loop, this line calculates the percentage of missing data for each column.
    print('{} - {}%'.format(col, pct_missing))


# In[7]:


df = df.dropna()


# In[8]:


# Data types for the columns

df.dtypes


# In[9]:


# Changing the data type of columns

df['budget'] = df['budget'].astype('int64')

df['gross'] = df['gross'].astype('int64')


# In[10]:


df


# In[11]:


# Creat year correct column
df['yearcorrect'] = df['released'].str.extract(pat = '([0-9]{4})').astype(int)


# In[12]:


df


# In[13]:


df = df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[14]:


pd.set_option('display.max_rows', None)


# In[15]:


# Drop any duplicates

df.drop_duplicates()


# In[16]:


# Finding correlations

# Hypothesis 1: Bugdet is correlated with gross revenue

# Scatter plot with budget vs gross

plt.scatter(x=df['budget'], y=df['gross'])

plt.title('Budget vs Gross Revenue')

plt.xlabel('Budget')

plt.ylabel('Gross')

plt.show()


# In[17]:


df.head()


# In[18]:


# Do they correlate? Using regression model

sns.regplot(x='budget', y='gross', data=df, scatter_kws={"color":"red"}, line_kws={"color":"blue"})


# In[23]:


# Looking at the correlations

numeric_df = df.select_dtypes(include=['number'])

correlation_matrix = numeric_df.corr()

print(correlation_matrix) 


# In[20]:


# High correlation between budget and gross


# In[25]:


sns.heatmap(correlation_matrix, annot = True)

plt.title('Correlation Matric for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movies Features')

plt.show()


# In[26]:


# Looking at Company

df.head()


# In[30]:


# Convert string into number

df_numerized = df

for col_name in df_numerized.columns:
    if(df_numerized[col_name].dtype == 'object'):
        df_numerized[col_name] = df_numerized[col_name].astype('category')
        df_numerized[col_name] = df_numerized[col_name].cat.codes
            
df_numerized


# In[31]:


correlation_matrix = df_numerized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot = True)

plt.title('Correlation Matric for Numeric Features')

plt.xlabel('Movie Features')

plt.ylabel('Movies Features')

plt.show()



# In[32]:


df_numerized.corr()


# In[33]:


correlation_mat = df_numerized.corr()

corr_pairs = correlation_mat.unstack()

corr_pairs


# In[34]:


sorted_pairs = corr_pairs.sort_values()

sorted_pairs


# In[35]:


high_corr = sorted_pairs[(sorted_pairs) > 0.5]

high_corr


# In[36]:


# Votes and budget have the highest correlation to gross earnings






