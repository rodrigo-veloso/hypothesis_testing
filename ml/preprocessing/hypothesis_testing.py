from os import stat
import pandas as pd
import numpy as np
import pingouin as pg
from pingouin import chi2_independence
import warnings
from scipy.stats import fisher_exact

class Tester:

    def __init__(self):
        """
        Constructor
        
    	Parameters
    	----------            
                    
    	Returns
    	-------
        Tester
        """
        self.selectors = {'compare_2_categorical':{'chi2':chi2_independence,
                                                                                'fisher_exact':fisher_exact}}

    def correlation_test(self, sample1, sample2, alpha = 0.05, alternative = 'two-sided', method = None, binary = ''):
        """
        Tests the null hypothesis that there is no correlation between quantitative samples (sample1,sample2)
        
    	Parameters
    	----------            
        sample1 : array_like
                  Array of sample data, must be quantitative data.
        sample2 : array_like
                  Array of sample data, must be quantitative data.
        alpha : float
                               level of significance (default = 0.05)
        test : string
                 correlation test to be applied
        binary : string
                      flag to identify if data is binary
                    
    	Returns
    	-------
        pd.DataFrame
        """
        sample1, sample2 = np.array(sample1), np.array(sample1) 
        np_types = [np.dtype(i) for i in [np.int32, np.int64, np.float32, np.float64]]
        sample1_dtypes, sample2_dtypes = sample1.dtype, sample2.dtype 
        if any([not t in np_types for t in [sample1_dtypes, sample2_dtypes]]):
            raise Exception('Non numerical variables... Try using categorical_test method instead.')

        report = ""
        if not method:
            if binary == 'yes':
                check = True
            elif binary == 'no':
                check = False
            else:
                check1 = self.check_binary(sample1)
                check2 = self.check_binary(sample2)
                check = check1 and check2

            if check:
                report += "Samples are binary, Pearson correlation is going to be applied (Point-biserial). "
                return self.correlation(sample1, sample2, 'pearson', alpha, report, alternative)
            else:
                check1 = self.normality_test(sample1).normal[0]
                check2 = self.normality_test(sample2).normal[0]
                check = check1 and check2
                if check:
                    report += "Samples have normal distribution. "
                    return self.correlation(sample1, sample2, 'pearson', alpha, report, alternative)
                else:
                    report += "Samples do not have normal distribution. "
                    return self.correlation(sample1, sample2, 'spearman', alpha, report, alternative)
        else:
            return self.correlation(sample1, sample2, method, alpha, report, alternative)

    def correlation(self, sample1, sample2, method, alpha, report, alternative):
        df = pg.corr(sample1, sample2, method = method, tail = alternative)
        result = True if df['p-val'][0] < alpha else False
        if result:
            report += "The alternative hypothesis is accepted, thus there is correlation between the samples. "
        else:
            report += "The null hypothesis is accepted, thus there is no correlation between the samples. " 
        report += "Significance level considered = {},  test applied = {}, p-value = {}, test statistic = {}. ".format(alpha, method, df['p-val'][0], df['r'][0])
        df['report'] = report
        return df

    def check_binary(self, col):
        for data in col:
            if data != 0 and data !=1:
                return False
        return True

    def categorical_test(self, data, sample1, sample2, alpha = 0.05, method = None):
        """
        Tests the null hypothesis that the categorical samples (sample1,sample2) are not dependent
        
    	Parameters
    	----------            
        data : pandas.DataFrame
                The dataframe containing the ocurrences for the test.
        sample1, sample2 : string
                The variables names for the Chi-squared test. Must be names of columns in ``data``.
        alpha : float
                level of significance (default = 0.05)
        method : string
                test to be applied
                    
    	Returns
    	-------
        pd.DataFrame
        """
        sample1_array, sample2_array = data[sample1], data[sample2]
        report = ""
        if not method:
            if len(sample1_array.unique()) == 2 and len(sample2_array.unique()) == 2:
                table = pd.crosstab(sample1_array, sample2_array)
                statistic, p_value = fisher_exact(table)
                self.categorical('fisher exact', statistic, p_value, alpha, report)
                return pd.DataFrame([('fisher exact', statistic, p_value)], columns = ['test', 'statistic', 'p_value'])
            else:
                if (data.groupby([sample1, sample2]).size() <= 5).sum():
                    warnings.warn("Warning: Algum valor esperado é menor do que 5. O teste pode ser inválido")
                expected, observed, stats = pg.chi2_independence(data, sample1, sample2)
                print("Expected Distribution")
                pg.print_table(expected.reset_index())
                print("Observed Distribution")
                pg.print_table(observed.reset_index())
                print("Statistics")
                p_value = stats.loc[stats['test'] == 'pearson']['pval'][0]
                statistic = stats.loc[stats['test'] == 'pearson']['chi2'][0]
                self.categorical('pearson chi-squared', statistic, p_value, alpha, report)
                return stats
        elif method == 'fisher':
            if not (len(sample1_array.unique()) == 2 and len(sample2_array.unique()) == 2):
                warnings.warn("Contigency table is not 2x2, Fisher exact cannot be used.")
                if (data.groupby([sample1, sample2]).size() <= 5).sum():
                    warnings.warn("Warning: Algum valor esperado é menor do que 5. O teste pode ser inválido")
                expected, observed, stats = pg.chi2_independence(data, sample1, sample2)
                print("Expected Distribution")
                pg.print_table(expected.reset_index())
                print("Observed Distribution")
                pg.print_table(observed.reset_index())
                print("Statistics")
                p_value = stats.loc[stats['test'] == 'pearson']['pval'][0]
                statistic = stats.loc[stats['test'] == 'pearson']['chi2'][0]
                self.categorical('pearson chi-squared', statistic, p_value, alpha, report)
                return stats
            else:
                table = pd.crosstab(sample1_array, sample2_array)
                statistic, p_value = fisher_exact(table)
                self.categorical('fisher exact', statistic, p_value, alpha, report)
                return pd.DataFrame([('fisher exact', statistic, p_value)], columns = ['test', 'statistic', 'p_value'])
        elif method == 'chi2':
            if (data.groupby([sample1, sample2]).size() <= 5).sum():
                    warnings.warn("Warning: Algum valor esperado é menor do que 5. O teste pode ser inválido")
            expected, observed, stats = pg.chi2_independence(data, sample1, sample2)
            print("Expected Distribution")
            pg.print_table(expected.reset_index())
            print("Observed Distribution")
            pg.print_table(observed.reset_index())
            print("Statistics")
            p_value = stats.loc[stats['test'] == 'pearson']['pval'][0]
            statistic = stats.loc[stats['test'] == 'pearson']['chi2'][0]
            self.categorical('pearson chi-squared', statistic, p_value, alpha, report)
            return stats
        else:
            raise Exception('Invalid method. Choose one of `fisher` or `chi2`.')
                
    def categorical(self, method, statistic, p_value, alpha, report):
        if p_value < alpha:
            report += "The null hypothesis is rejected, thus there is evidence of dependency between the samples"
        else:
            report += "The null hypothesis is not rejected, thus there is no evidence of dependency between the samples"
        report += " \nTest applied = {} \nSignificance level = {} \np-value = {} \nTest Statistic = {}. ".format(method, alpha, p_value, statistic)
        print(report)
        print()
        

    def normality_test(self, sample, alpha = 0.05, method = 'shapiro'):
        """
        Tests the null hypothesis that the data was drawn from a normal distribution
        
    	Parameters
    	----------            
        sample : array_like
                  Array of sample data.
        alpha : float
                               level of significance (default = 0.05)
        test : string
                 normality test to be applied
                    
    	Returns
    	-------
        pd.DataFrame
        """
        report = ""
        sample = np.array(sample)
        np_types = [np.dtype(i) for i in [np.int32, np.int64, np.float32, np.float64]]
        sample_dtypes = sample.dtype
        if any([not t in np_types for t in [sample_dtypes]]):
            raise Exception('Non numerical variables... Try using categorical_test method instead.')
        df = pg.normality(sample)
        result = True if df['pval'][0] >= alpha else False
        if result:
            report += "The null hypothesis is accepted, thus the data was  drawn from a normal distribution"
        else:
            report += "The alternative hypothesis is accepted, thus the data was not  drawn from a normal distribution. "
        report += "Significance level considered = {},  test applied = {}, p-value = {}, test statistic = {}".format(alpha, method, df['pval'][0], df['W'][0])
        df['report'] = report
        return df
    
