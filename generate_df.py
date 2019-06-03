
# coding: utf-8

# In[5]:


import AnuLibrary, writeFact
from IPython.display import HTML
# path = '/home/kishori/rule1E_tmp/2.1/'
relation_df = AnuLibrary.create_dataframe('hindi_dep_parser_original.dat') 
relation_df

HTML(relation_df.to_html(classes='table table-condensed'))

