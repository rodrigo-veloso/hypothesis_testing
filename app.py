
import streamlit as st
import matplotlib.pyplot as plt

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
st.write(""" Tests whether a data sample has a Gaussian distribution. The Shapiro-Wilk Test is set as default. One can also apply D’Agostino’s $K^2$ Test""")

st.write("""### Default test example""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    t = Tester()
    results = t.normality_test(df['Age'])
    sns_plot = sns.distplot(df['Age'])
    print(results)
    print("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.pyplot(sns_plot.figure)
st.write(results)
st.write("p-value = {}, significance = {}".format(results.p_value, results.significance))

plt.clf()

st.write("""### D’Agostino’s $K^2$  test example""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    t = Tester()
    results = t.normality_test(df['Age'][:30], test = 'dagostinos')
    sns_plot = sns.distplot(df['Age'][:30])
    print(results)
    print("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.pyplot(sns_plot.figure)
st.write(results)
st.write("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.write("""## Dependency test between two categorical variables""")
st.write(""" Tests whether two categorical variables are related or independent. Using the default configuration: If all the cells of contigency table have values above 5, Chi-Squared test is used, otherwise the normality condition is not satisfied for Chi-Squared and dimensions of contigency table are checked. If the contigency table is 2x2 than both Fisher-Exact test and Chi-Squared test are applied, else Chi-Squared test is applied even though the normality condition is not satisfied. One can also specifies what test should be applied using the test parameter.""")

st.write("""### Normality condition satisfied for Chi-Squared example""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    t = Tester()
    results = t.categorical_test(df['Survived'], df['Sex'])
    print(results)
    print("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.write(results)
st.write("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.write("""### Normality condition not satisfied for Chi-Squared example""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    t = Tester()
    results = t.categorical_test(df['Survived'][:20], df['Sex'][:20])
    print(results)
    print(results.chi2)
    print("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.write(results)
st.write(results.chi2)
st.write("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.write("""### Normality condition not satisfied for Chi-Squared and contigency table not 2x2 example""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    t = Tester()
    results = t.categorical_test(df['Survived'][:20], df['Pclass'][:20])
    print(results)
    print("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.write(results)
st.write("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.write("""### Using just a specific test example (no checks are made)""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    t = Tester()
    results = t.categorical_test(df['Survived'][:20], df['Pclass'][:20], test = 'chi2')
    print(results)
    print("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.write(results)
st.write("p-value = {}, significance = {}".format(results.p_value, results.significance))

st.write("""## Correlation test between two quantitative variables""")
st.write(""" Tests whether two quantitative variables have a linear relationship. Using the default configuration: If both variables are binary, Pearson test is applied (Point-biserial). If variables are not binary and have a normal distribution, Pearson test is also applied, else Spearmen test is applied. One can also specifies what test should be applied using the test parameter.""")

st.write("""### Binary variables example""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    df = p.categ_encoding(df)
    t = Tester()
    results = t.correlation_test(df['Survived'][:20], df['Sex_female'][:20], method='pearson')
    print(results)

st.write(results)
st.write(results['report'][0])

st.write("""### Normally distributed example""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    df['2xAge'] = df['Age']*2
    t = Tester()
    results = t.correlation_test(df['2xAge'][:20], df['Sex'][:20])
    print(results)

st.write(results)

st.write("""### Normally distributed example""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    df['2xAge'] = df['Age']*2
    t = Tester()
    results = t.correlation_test(df['2xAge'][:20], df['Age'][:20])
    print(results)

st.write(results)

st.write("""### Not normally distributed example""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    df['2xAge'] = df['Age']*2
    t = Tester()
    results = t.correlation_test(df['2xAge'], df['Age'].tolist())
    print(results)

st.write(results)

st.write("""### Using just a specific test example (no checks are made)""")

with st.echo():
    p = Preprocessing()
    df = p.clean_data(df)
    df['2xAge'] = df['Age']*2
    t = Tester()
    t = Tester()
    results = t.correlation_test(df['2xAge'], df['Age'], method = 'kendall')
    print(results)

st.write(results)
