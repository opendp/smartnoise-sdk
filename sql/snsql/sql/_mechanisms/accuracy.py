import math
from os import name
from typing import Tuple
from snsql._ast.ast import Query
from snsql._ast.expression import NamedExpression
from snsql._ast.expressions.numeric import ArithmeticExpression
from snsql.sql.privacy import Privacy
from opendp.accuracy import gaussian_scale_to_accuracy
from snsql.sql._mechanisms import *

class Accuracy:
    """
    Routines to compute the error bounds for summary statistic at specified
    alpha.  These are hard-coded to the Gaussian mechanism we use, and will need
    to be generalized if we allow pluggable mechanisms.

    All formulas from: https://github.com/opendp/smartnoise-sdk/blob/main/papers/DP_SQL_budget.pdf
    """
    def __init__(self, query: Query, subquery: Query, privacy: Privacy):
        """
        Detection of formulas happens only once per query
        """
        self.privacy = privacy
        self.max_contrib = query.max_ids # always take from Query
        
        detect = DetectFormula(query, subquery)
        self.properties = [detect.detect(ne) for ne in query.xpath("/Query/Select//NamedExpression")]
    """
    Each formula takes an alpha, properties, and row.
    The properties argument gives the locations and sensitivities
     of each column from the subquery that is used to compute this
     output column.  The properties are populated at detect time.
    """
    def count(self, *ignore, alpha: float, properties={}, row:Tuple=None):
        return self.error_range(alpha=alpha, stat='count', sensitivity=1)
    def threshold(self, *ignore, alpha: float, properties={}, row:Tuple=None):
        return self.error_range(alpha=alpha, stat='threshold', sensitivity=1)
    def sum(self, *ignore, alpha: float, properties={}, row:Tuple=None):
        return self.error_range(alpha=alpha, stat='sum', sensitivity=properties["sensitivity"]["sum"])
    def _mean(self, *ignore, alpha:float, sigma_1:float, sigma_2:float, n:float, sum_val:float):
        # mean, variance, and stddev all use this formula
        shift = math.sqrt(2 * math.log(4/alpha)) 
        if n <= 2 * shift * sigma_1:
            return None
        else:
            left = (shift * sigma_2)/n
            right = (2 * shift * math.fabs(sum_val) * sigma_1 + 4 * math.log(4/alpha) * sigma_1 * sigma_2) 
            right = right / (n * n)
            return left + right

    def mean(self, *ignore, alpha: float, properties={}, row: Tuple):
        n_idx = properties['columns']['count']
        sum_idx = properties['columns']['sum']
        n = row[n_idx]
        sum_val = row[sum_idx]
        scale = self.gauss_scale(stat='count', sensitivity=1)
        sum_sens = properties['sensitivity']['sum']
        scale_sum = self.gauss_scale(stat='sum', sensitivity=sum_sens)
        return self._mean(alpha=alpha, sigma_1=scale, sigma_2=scale_sum, n=n, sum_val=sum_val)

    def variance(self, *ignore, alpha: float,  properties={}, row: Tuple):
        alpha = 2.0/3.0 * alpha
        n_idx = properties['columns']['count']
        sum_idx = properties['columns']['sum']
        sum_s_idx = properties['columns']['sum_of_squares']
        n = row[n_idx]
        sum_val = row[sum_idx]
        sum_s_val = row[sum_s_idx]
        scale_count = self.gauss_scale(stat='count', sensitivity=1)
        sum_sens = properties['sensitivity']['sum']
        scale_sum = self.gauss_scale(stat='sum', sensitivity=sum_sens)
        scale_sum_s = self.gauss_scale(stat='sum', sensitivity=(sum_sens*sum_sens))
        f1 = self._mean(alpha=alpha, sigma_1=scale_count, sigma_2=scale_sum_s, n=n, sum_val=sum_s_val)
        f2 = self._mean(alpha=alpha, sigma_1=scale_count, sigma_2=scale_sum, n=n, sum_val=sum_val)
        if f1 and f2:
            return f1 + f2 * (f2 + ((2*sum_val)/n))
        else:
            return None
    def stddev(self, *ignore, alpha: float,  properties={}, row:Tuple):
        v = self.variance(alpha=alpha, properties=properties, row=row)
        if v:
            return math.sqrt(v)
        else:
            return None
    def _mech(self, *ignore, stat: str, sensitivity):
        if not isinstance(sensitivity, (int, float)):
            raise ValueError(f"Sensitivity must be int or float.  Got {type(sensitivity)}")
        t = 'int' if isinstance(sensitivity, int) else 'float'
        mech_class = self.privacy.mechanisms.get_mechanism(sensitivity, stat, t)
        return mech_class(self.privacy.epsilon, delta=self.privacy.delta, sensitivity=sensitivity, max_contrib=self.max_contrib)
    def scale(self, *ignore, stat: str, sensitivity):
        mech = self._mech(stat=stat, sensitivity=sensitivity)
        return mech.scale
    def gauss_scale(self, *ignore, stat: str, sensitivity):
        mech = Gaussian(self.privacy.epsilon, delta=self.privacy.delta, sensitivity=sensitivity, max_contrib=self.max_contrib)
        return mech.scale
    def error_range(self, *ignore, alpha: float, stat: str, sensitivity):
        mech = self._mech(stat=stat, sensitivity=sensitivity)
        return mech.accuracy(alpha)
    def accuracy(self, *ignore, row:Tuple, alpha:float):
        """
        Returns a tuple of the same size as the output row, with +/-
         accuracy for the supplied alpha.  This method will be called
         once per output row, per alpha.
        Returns 'None' if accuracy is not relevant or available.
        Requires a row argument with a tuple of the same size as the subquery result.
        """
        out_row = []
        for p in self.properties:
            if p is None:
                out_row.append(None)
            else:
                stat = p["statistic"]
                r = None
                if stat == "count":
                    r = self.count(alpha=alpha, properties=p, row=row)
                elif stat == "threshold":
                    r = self.threshold(alpha=alpha, properties=p, row=row)
                elif stat == "sum":
                    r = self.sum(alpha=alpha, properties=p, row=row)
                elif stat == "mean":
                    r = self.mean(alpha=alpha, properties=p, row=row)
                elif stat == "variance":
                    r = self.variance(alpha=alpha, properties=p, row=row)
                elif stat == "stddev":
                    r = self.stddev(alpha=alpha, properties=p, row=row)
                out_row.append(r)
        return tuple(out_row)
            

class DetectFormula:
    def __init__(self, query: Query, subquery: Query):
        self.query = query
        self.subquery = subquery
    def get_index(self, query: Query, cname: str) -> int:
        """
        Helper method, gets the column index of a named output column
        """
        namedExpressions = query.xpath("/Query/Select//NamedExpression")
        for idx in range(len(namedExpressions)):
            if namedExpressions[idx].name == cname:
                return idx
        return -1
    def get_sensitivity(self, node: NamedExpression):
        """
        Helper method, gets the sensitivity of the given NamedExpression
        """
        source_path = "@m_symbol/@expression//TableColumn"
        source = node.xpath_first(source_path)
        if source:
            return source.sensitivity()
        else:
            return None
    def count(self, node: NamedExpression):
        """
        Use XPath to match any output column like (count_age)
         where count_age is COUNT(age) in the subquery
        """
        cname = node.xpath_first('/NamedExpression/NestedExpression/Column/@name')
        if cname:
            cname = cname.value
            source_path = f"/Query/Select//NamedExpression[@name='{cname}']"
            source_idx = self.get_index(self.subquery, cname)
            source = self.subquery.xpath_first(source_path)
            source_path = "@expression/AggFunction[@name='COUNT']"
            if source.xpath_first(source_path):
                tc_source_path = "@m_symbol//TableColumn"
                ac_source_path = "@m_symbol//AllColumns"
                if source.xpath_first(tc_source_path) or source.xpath_first(ac_source_path):
                    return {
                        'statistic': 'threshold' if cname == 'keycount' else 'count',
                        'columns':  {
                            'count': source_idx,
                        },
                        'sensitivity': {
                            'count': 1
                        }
                    }
        return None
    def sum(self, node: NamedExpression):
        """
        Use XPath to match any output column like (sum_age)
         where count_age is SUM(age) in the subquery
        """
        cname = node.xpath_first('/NamedExpression/NestedExpression/Column/@name')
        if cname:
            cname = cname.value
            source_path = f"/Query/Select//NamedExpression[@name='{cname}']"
            source_idx = self.get_index(self.subquery, cname)
            source = self.subquery.xpath_first(source_path)
            source_path = "@expression/AggFunction[@name='SUM']"
            if source.xpath_first(source_path):
                sens = self.get_sensitivity(source)
                if sens:
                    return {
                        'statistic': 'sum',
                        'columns':  {
                            'sum': source_idx,
                        },
                        'sensitivity': {
                            'sum': sens
                        }
                    }
        return None
    def mean(self, node:NamedExpression):
        """
        Use XPath to match any output column like (sum_age / count_age)
         where count_age is COUNT(age) in the subquery and sum_age is SUM(age)
        """
        cnames = node.xpath("/NamedExpression/NestedExpression/NestedExpression/ArithmeticExpression[@op='/']//Column/@name")
        if len(cnames) == 2:
            l_name, r_name = [c.value for c in cnames]
            l_source_path = f"/Query/Select//NamedExpression[@name='{l_name}']"
            l_idx = self.get_index(self.subquery, l_name)
            l_source = self.subquery.xpath_first(l_source_path)
            r_source_path = f"/Query/Select//NamedExpression[@name='{r_name}']"
            r_idx = self.get_index(self.subquery, r_name)
            r_source = self.subquery.xpath_first(r_source_path)
            sens = self.get_sensitivity(r_source)

            if (
                l_source.xpath("AggFunction[@name='SUM']") and 
                r_source.xpath("AggFunction[@name='COUNT']") and
                l_source.xpath_first("//Column").name == r_source.xpath_first("//Column").name
            ):
                    return {
                        'statistic': 'mean',
                        'columns':  {
                            'sum': l_idx,
                            'count': r_idx
                        },
                        'sensitivity': {
                            'sum': sens,
                            'count': 1
                        }
                    }
        return None
    def _check_var_formula(self, node:ArithmeticExpression):
        """
        Use XPath to match an expression that looks like
         ( ( sum_alias_0xd539 / count_alias_0xd539 ) - ( sum_age / count_age ) * ( sum_age / count_age ) )
         where the sum_alias is SUM(age*age), and sum_age and count_age are as in mean formula
        """
        subtract = node
        left = subtract.left.xpath_first("/NestedExpression/ArithmeticExpression[@op='/']")
        if left:
            l_cnames = left.xpath("//Column/@name")
            if len(l_cnames) == 2:
                right = subtract.right.xpath_first("/ArithmeticExpression[@op='*']")
                if right:
                    rightleft = right.left.xpath_first("/NestedExpression/ArithmeticExpression[@op='/']")
                    rightright = right.right.xpath_first("/NestedExpression/ArithmeticExpression[@op='/']")
                    rl_cnames = rightleft.xpath("//Column/@name")
                    rr_cnames = rightright.xpath("//Column/@name")
                    if (
                        len(rl_cnames) == 2 and 
                        len(rr_cnames) == 2 and 
                        all([rl.value == rr.value for rl, rr in zip(rl_cnames, rr_cnames)])
                    ):
                        sum_of_squares = self.subquery.xpath_first(f"/Query/Select//NamedExpression[@name='{l_cnames[0].value}']")
                        sum_s_idx = self.get_index(self.subquery, l_cnames[0].value)
                        sum_s_sens = self.get_sensitivity(sum_of_squares)
                        count_of_squares = self.subquery.xpath_first(f"/Query/Select//NamedExpression[@name='{l_cnames[1].value}']")
                        sum_col = self.subquery.xpath_first(f"/Query/Select//NamedExpression[@name='{rl_cnames[0].value}']")
                        sum_idx = self.get_index(self.subquery, rl_cnames[0].value)
                        sum_sens = self.get_sensitivity(sum_col)
                        count_col = self.subquery.xpath_first(f"/Query/Select//NamedExpression[@name='{rl_cnames[1].value}']")
                        count_idx = self.get_index(self.subquery, rl_cnames[1].value)
                        # we don't need to check right side, since we verified source is the same as left
                        if (
                            sum_of_squares and 
                            count_of_squares and 
                            sum_col and 
                            count_col
                        ):
                            if (
                                sum_of_squares.xpath("AggFunction[@name='SUM']/ArithmeticExpression[@op='*']") and 
                                count_of_squares.xpath("AggFunction[@name='COUNT']")
                            ):
                                sum_s_cols = sum_of_squares.xpath("//Column/@name")
                                count_s_cols = count_of_squares.xpath("//Column/@name")
                                sum_cols = sum_col.xpath("//Column/@name")
                                count_cols = count_col.xpath("//Column/@name")
                                if (
                                    len(sum_s_cols) == 2 and 
                                    len(count_s_cols) == 1 and 
                                    len(sum_cols) == 1 and
                                    len(count_cols) == 1 and
                                    sum_cols[0].value == count_cols[0].value and
                                    sum_s_cols[0].value == sum_s_cols[1].value and 
                                    all([s.value == count_s_cols[0].value for s in sum_s_cols])
                                    and sum_cols[0].value == sum_s_cols[0].value
                                ):
                                    return {
                                        'statistic': 'variance',
                                        'columns': {
                                            'sum_of_squares': sum_s_idx,
                                            'sum': sum_idx,
                                            'count': count_idx
                                        },
                                        'sensitivity': {
                                            'sum_of_squares': sum_s_sens,
                                            'sum': sum_sens,
                                            'count': 1
                                        }
                                    }
        return None
    def variance(self, node:NamedExpression):
        subtract = node.xpath_first("/NamedExpression/NestedExpression/ArithmeticExpression[@op='-']")
        if subtract:
            return self._check_var_formula(subtract)
        else:
            return None
    def stddev(self, node:NamedExpression):
        # same formula as variance, but inside SQRT(...)
        sqrt = node.xpath_first("NestedExpression/MathFunction[@name='SQRT']")
        if sqrt:
            retval = self._check_var_formula(sqrt.expression)
            if retval:
                retval['statistic'] = 'stddev'
            return retval
        else:
            return None
    def detect(self, node: NamedExpression):
        """
        Checks a single NamedExpression against all known formulas
         for which accuracy is available.  Returns a properties bag for the
         formula if there is a match.
        """
        p = self.count(node)
        if p:
            return p
        p = self.sum(node)
        if p:
            return p
        p = self.mean(node)
        if p:
            return p
        p = self.variance(node)
        if p:
            return p
        p = self.stddev(node)
        if p:
            return p
        return None
