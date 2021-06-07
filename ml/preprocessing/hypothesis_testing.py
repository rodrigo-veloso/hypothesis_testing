import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.stats import normaltest

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
                                                                 'dagostinos':normaltest}}

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
        result = True if p_value > significance else False
        if result:
            report += "The null hypothesis is accepted, thus the data was  drawn from a normal distribution"
        else:
            report += "The alternative hypothesis is accepted, thus the data was not  drawn from a normal distribution. "
        report += "Significance level considered = {},  test applied = {}, p-value = {}, test statistic = {}".format(significance, test, p_value, statistic)
        return TestResult(statistic, p_value, significance, test, result, report)
    
