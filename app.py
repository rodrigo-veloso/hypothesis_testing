
import streamlit as st

st.write("""# Hypothesis test""")

st.write("""### Examples of hypothesis tests using the hermione framework from A3Data""")
st.write("""### https://github.com/A3Data/hermione""")

st.image('vertical_logo.png', width = 600)

st.write("""## Import:""")

st.write("""### All necessary modules""")

with st.echo():
    import pandas as pd
    import numpy as np
    from ml.data_source.spreadsheet import Spreadsheet
    from ml.preprocessing.hypothesis_testing import Tester
    from ml.preprocessing.preprocessing import Preprocessing
    import seaborn as sns

st.write("""### The data""")
with st.echo():
    df = Spreadsheet().get_data('train.csv',columns=['Survived','Pclass','Sex','Age'])

st.write(df.head(5))

st.write("""## Normality Test""")
with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    t = Tester()
    results = t.normality_test(df['Age'], test='shapiro-wilk')
    sns_plot = sns.distplot(df['Age'])
    print(results)
    print("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.pyplot(sns_plot.figure)
st.write(results)
st.write("p-value = {}, significance = {}".format(results.p_value, results.significance))
