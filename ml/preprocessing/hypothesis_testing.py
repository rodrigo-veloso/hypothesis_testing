import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau

class TestResult:

    def __init__(self, statistic, p_value, significance, test, result, report):
        """ 
        Constructor

        Parameters
        ----------
            statistic :          float
                                     test statistics
            p-value :           float
                                     test's p-value
            significance :   float
                                     significance level considered
            test :                 string
                                     test applied
            result :             bool
                                     test result, True if alternative hypothesis is accepted
            report :             string
                                     sumary of the test results

        Returns   
        ----------
            TestResult
        """                 

        self.statistic = statistic
        self.p_value = p_value
        self.significance = significance
        self.test = test
        self.result = result
        self.report = report

    def __str__(self):
        return "statistic = {}\n p-value = {}\n sigificance = {}\n test = {}\n result = {}\n report: {}".format(self.statistic,self.p_value,self.significance,self.test,self.result,self.report)

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
        self.selectors = {'normality_test':{'shapiro-wilk':shapiro,
                                                                 'dagostinos':normaltest},
                                    'compare_2_categorical':{'chi2':chi2_contingency,
                                                                                'fisher_exact':fisher_exact},
                                    'correlation_test':{'pearson':pearsonr,
                                                                   'spearman':spearmanr,
                                                                   'kendalltau':kendalltau}}

    def correlation_test(self, col1, col2, significance = 0.05, test = 'default', binary = ''):
        """
        Tests the null hypothesis that there is no correlation between quantitative samples (col2,col2)
        
    	Parameters
    	----------            
        col1 : array_like
                  Array of sample data, must be quantitative data.
        col1 : array_like
                  Array of sample data, must be quantitative data.
        significance : float
                               level of significance (default = 0.05)
        test : string
                 correlation test to be applied
        binary : string
                      flag to identify if data is binary
                    
    	Returns
    	-------
        TestResult
        """
        report = ""
        if test == 'default':
            if binary == 'yes':
                check = True
            elif binary == 'no':
                check = False
            else:
                check1 = self.check_binary(col1)
                check2 = self.check_binary(col2)
                check = check1 and check2

            if check:
                report += "Samples are binary, Pearson correlation is going to be applied (Point-biserial). "
                return self.correlation(col1, col2, 'pearson', significance, report)
            else:
                check1 = self.normality_test(col1).result
                check2 = self.normality_test(col2).result
                check = check1 and check2
                if check:
                    report += "Samples have normal distribution. "
                    return self.correlation(col1, col2, 'pearson', significance, report)
                else:
                    report += "Samples do not have normal distribution. "
                    return self.correlation(col1, col2, 'spearman', significance, report)
        else:
            return self.correlation(col1, col2, test, significance, report)

    def correlation(self, col1, col2, test, significance, report):
        statistic, p_value = self.selectors['correlation_test'][test](col1, col2)
        result = True if p_value < significance else False
        if result:
            report += "The alternative hypothesis is accepted, thus there is correlation between the samples. "
        else:
            report += "The null hypothesis is accepted, thus there is no correlation between the samples. " 
        report += "Significance level considered = {},  test applied = {}, p-value = {}, test statistic = {}. ".format(significance, test, p_value, statistic)
        return TestResult(statistic, p_value, significance, test, result, report)

    def check_binary(self, col):
        for data in col:
            if data != 0 and data !=1:
                return False
        return True

    def categorical_test(self, col1, col2, significance = 0.05, test = 'default'):
        """
        Tests the null hypothesis that the categorical samples (col1,col2) are not dependent
        
    	Parameters
    	----------            
        col1 : array_like
                  Array of sample data, must be categorical data.
        col1 : array_like
                  Array of sample data, must be categorical data.
        significance : float
                               level of significance (default = 0.05)
        test : string
                 test to be applied
                    
    	Returns
    	-------
        TestResult
        """

        contigency_table = pd.crosstab(col1, col2)
        normality_condition = True
        report = ""
        if test == 'default':
            for row in contigency_table.iloc:
                for data in row:
                    if data <= 5:
                        normality_condition = False
                        report += "Normality condition not satisfied (observed and expected frequency in some cell of contigency table <= 5). Chi2 may not be valid. "
                        break
                else:
                    continue
                break
            if  normality_condition:
                report += "Normality condition is satisfied. "
                return self.categorical(contigency_table, 'chi2', significance, report)
            else:
                shape = np.shape(contigency_table)
                if shape[0] == 2 and shape[1] == 2:
                    fisher = self.categorical(contigency_table, 'fisher_exact', significance, report)
                    fisher.chi2 = self.categorical(contigency_table, 'chi2', significance, report)
                    fisher.report += "To view chi2 results check the chi2 attribute (TestResult.chi2)"
                    return fisher
                else:
                    report += "Contigency table is not 2x2, Fisher exact cannot be used. "
                    return self.categorical(contigency_table, 'chi2', significance, report)
        else:
            return self.categorical(contigency_table, test, significance, report)
                
    def categorical(self, contigency_table, test, significance, report):
        output = self.selectors['compare_2_categorical'][test](contigency_table)
        statistic, p_value = output[0], output[1]
        result = True if p_value < significance else False
        if result:
            report += "The alternative hypothesis is accepted, thus there is a dependency between the samples. "
        else:
            report += "The null hypothesis is accepted, thus the two samples are independent. " 
        report += "Significance level considered = {},  test applied = {}, p-value = {}, test statistic = {}. ".format(significance, test, p_value, statistic)
        return TestResult(statistic, p_value, significance, test, result, report)
        

    def normality_test(self, col1, significance = 0.05, test = 'shapiro-wilk'):
        """
        Tests the null hypothesis that the data was drawn from a normal distribution
        
    	Parameters
    	----------            
        col1 : array_like
                  Array of sample data.
        significance : float
                               level of significance (default = 0.05)
        test : string
                 normality test to be applied
                    
    	Returns
    	-------
        TestResult
        """
        report = ""
        if test == 'dagostinos' and len(col1) < 20:
            report += "Warning!!! Test only valid for n>=20 (n = number of samples in data). "

        statistic, p_value = self.selectors['normality_test'][test](col1)
        result = True if p_value >= significance else False
        if result:
            report += "The null hypothesis is accepted, thus the data was  drawn from a normal distribution"
        else:
            report += "The alternative hypothesis is accepted, thus the data was not  drawn from a normal distribution. "
        report += "Significance level considered = {},  test applied = {}, p-value = {}, test statistic = {}".format(significance, test, p_value, statistic)
        return TestResult(statistic, p_value, significance, test, result, report)
    
