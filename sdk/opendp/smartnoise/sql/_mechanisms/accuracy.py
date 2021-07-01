import math
from os import name
from typing import List, Tuple
from opendp.smartnoise._ast.ast import Query
from opendp.smartnoise._ast.expression import NamedExpression
from opendp.smartnoise.sql.privacy import Privacy
from scipy.stats import norm

class Accuracy:
    """
    Routines to compute the error bounds for summary statistic at specified
    alpha.  These are hard-coded to the Gaussian mechanism we use, and will need
    to be generalized if we allow pluggable mechanisms.
    """
    def __init__(self, query: Query, subquery: Query, privacy : Privacy):
        self.privacy = privacy
        self.max_contrib = query.max_ids
        self.base_scale = self.scale(sensitivity=1.0)
        
        detect = DetectFormula(query, subquery)
        self.properties = [detect.detect(ne) for ne in query.xpath("/Query/Select//NamedExpression")]

    def count(self, *ignore, alpha: float, properties={}, row:Tuple=None):
        sigma = self.scale(sensitivity=1)
        return self.percentile(percentile=1 - alpha, sigma=sigma)
    def sum(self, *ignore, alpha: float, properties={}, row:Tuple=None):
        sigma = self.scale(sensitivity=properties["sensitivity"]["sum"])
        return self.percentile(percentile=1 - alpha, sigma=sigma)
    def mean(self, *ignore, alpha: float, properties={}, row: Tuple):
        n_idx = properties['columns']['count']
        sum_idx = properties['columns']['sum']
        n = row[n_idx]
        sum_val = row[sum_idx]
        sigma = self.scale(sensitivity=1)
        sum_sens = properties['sensitivity']['sum']
        sigma_sum = self.scale(sensitivity=sum_sens)
        shift = math.sqrt(2 * math.log(4/alpha)) 
        if n <= 2 * shift * sigma:
            return None
        else:
            left = (shift * sigma_sum)/n
            right = (2 * shift * math.fabs(sum_val) * sigma + 4 * math.log(4/alpha) * sigma * sigma_sum) / (n * n)
            return right + left
    def variance(self, *ignore, alpha: float,  properties={}, row: Tuple):
        return None
    def stddev(self, *ignore, alpha: float,  properties={}, row:Tuple):
        return None
    def scale(self, *ignore, sensitivity: float):
        sigma = (math.sqrt(math.log(1/self.privacy.delta)) + math.sqrt(math.log(1/self.privacy.delta) + self.privacy.epsilon)) / (math.sqrt(2) * self.privacy.epsilon)
        return sigma * self.max_contrib * sensitivity
    def percentile(self, *ignore, percentile: float, sigma: float):
        dist = norm(0, sigma)
        right = (1.0 + percentile) / 2
        return dist.ppf(right)

class DetectFormula:
    def __init__(self, query: Query, subquery: Query):
        self.query = query
        self.subquery = subquery
    def get_index(self, query: Query, cname: str) -> int:
        namedExpressions = query.xpath("/Query/Select//NamedExpression")
        for idx in range(len(namedExpressions)):
            if namedExpressions[idx].name == cname:
                return idx
        return -1
    def get_sensitivity(self, node: NamedExpression):
        source_path = "@m_symbol/@expression//TableColumn"
        source = node.xpath_first(source_path)
        if source:
            return source.sensitivity()
        else:
            return None
    def count(self, node: NamedExpression):
        # expects a named expression
        cname = node.xpath_first('/NamedExpression/NestedExpression/Column/@name')
        if cname:
            cname = cname.value
            source_path = f"/Query/Select//NamedExpression[@name='{cname}']"
            source_idx = self.get_index(self.subquery, cname)
            source = self.subquery.xpath_first(source_path)
            source_path = "@expression/AggFunction[@name='COUNT']"
            if source.xpath_first(source_path):
                source_path = "@m_symbol/@expression//TableColumn"
                source = source.xpath_first(source_path)
                if source:
                    return {
                        'statistic': 'count',
                        'columns':  {
                            'count': source_idx,
                        },
                        'sensitivity': {
                            'count': 1
                        }
                    }
        return None
    def sum(self, node: NamedExpression):
        # expects a named expression
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

            if l_source.xpath("AggFunction[@name='SUM']") and r_source.xpath("AggFunction[@name='COUNT']"):
                if l_source.xpath_first("//Column").name == r_source.xpath_first("//Column").name:
                    return {
                        'statistic': 'mean',
                        'columns':  {
                            'sum': r_idx,
                            'count': l_idx
                        },
                        'sensitivity': {
                            'sum': sens,
                            'count': 1
                        }
                    }
        return None
    def detect(self, node: NamedExpression):
        p = self.count(node)
        if p:
            return p
        p = self.sum(node)
        if p:
            return p
        p = self.mean(node)
        if p:
            return p
        return None
