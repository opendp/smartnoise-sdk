
import random
from opendp.smartnoise.ast.tokens import *
from opendp.smartnoise.ast.expression import *
from opendp.smartnoise.ast import tokens
from opendp.smartnoise.ast.expressions import *
from opendp.smartnoise.sql import PandasReader, PrivateReader
# from opendp.smartnoise.sql.private_reader import PrivateReaderOptions
from opendp.smartnoise.evaluation.params._learner_params import LearnerParams



def columns_to_select(ep: LearnerParams):
    columns = ep.columns
    # return ['UserId', 'Role', 'Usage']
    return columns

# def QuerytoAST(query, meta, data):
#     reader = PandasReader(meta, data)
#     # prOptions = PrivateReaderOptions(censor_dims = False)
#     # private_reader = PrivateReader(meta, reader, epsilon = 1, options=prOptions)    
#     private_reader = PrivateReader(meta, reader, self.epsilon)
#     try:
#         ast = private_reader.parse_query_string(query) 
#     except:
#         return
#     return ast

def FindNode(ast):
    # ne = ast.select.namedExpressions 
    node_list = ast.select.find_nodes(SqlExpr)
    for n in node_list:
        if n is not None:
            n.set_parent()
    return node_list

def AddNumericLiteral(ast, action):
    nums = ast.find_nodes(Literal)
    if len(nums) > 0:
        num = random.choice(nums)
        num.value = num.value + 1000
        num.text = str(num.value)
    return str(ast)

def SubstractNumericLiteral(ast, action):
    nums = ast.find_nodes(Literal)
    if len(nums) > 0:
        num = random.choice(nums)
        num.value = num.value - 1000
        num.text = str(num.value)
    return str(ast)

def ReplaceAggFunction(ast, action):
    agg = action['aggfunc']
    nums = ast.find_nodes(AggFunction)
    if len(nums) > 0:
        num = random.choice(nums)
        num.name = agg
    return str(ast)

def ReplaceColumn(ast, action):
    nums = ast.find_nodes(Column)
    if len(nums) > 0 and action['column']!='Allcolumns':
        num = random.choice(nums)
        num.name = action['column']
    return str(ast)

def ReplaceOP(ast, action):
    nums = ast.find_nodes(ArithmeticExpression)
    if len(nums) > 0:
        num = random.choice(nums)
        num.op = action['op']
    return str(ast)


def DeleteNode(ast, action):
    node_list = FindNode(ast)
    if len(node_list) > 0 :
        node = random.choice(node_list)
        delete(ast, node)         
    return str(ast)

def delete(ast, todel):
    if not hasattr(todel, 'parent'):
        ast.select.namedExpressions.seq.remove(todel)
    elif not hasattr(todel.parent, 'left'):
        delete(ast, todel.parent)
    else:
        if todel == todel.parent.right:
            todel.parent.right = None
            todel.parent.op = None
        elif todel == todel.parent.left:
            todel.parent.left = None
            todel.parent.op = None     

def InsertNamedExpression(ast, action):
    column = action['column']
    # construct expression to be inserted
    if column == 'Allcolumns':
        addexpression = sql.AllColumns(None)
    else:
        addexpression = sql.Column(column)
    addexpression = sql.AggFunction(action['aggfunc'], None, addexpression)
    addnamedexpression = NamedExpression(None, addexpression)
    ast.select.namedExpressions.seq.append(addnamedexpression)    
    return ast 

def InsertArithmeticExpression(ast, action):
    node_list = FindNode(ast)
    agg = action['aggfunc']
    op = action['op']
    column = action['column']

    # decide if it's a valid place to insert
    if len(node_list) > 0:
        toinsert = random.choice(node_list)
        # construct expression to be inserted
        if column == 'Allcolumns':
            addexpression = sql.AllColumns(None)
        else:
            addexpression = sql.Column(column)
        expr_tbi = sql.AggFunction(agg, None, addexpression)
        if hasattr(toinsert, 'parent') and isinstance(toinsert.parent, NamedExpression):
            addAithmeticExpression = numeric.ArithmeticExpression(expr_tbi, op, toinsert)
            expr_tbi = NestedExpression(addAithmeticExpression)
            toinsert.parent.expression = expr_tbi
        elif not hasattr(toinsert, 'parent'):
            ast.select.namedExpressions.seq.append(expr_tbi)
    return ast


def InsertCaseExpression(ast, action):
    node_list = FindNode(ast)
    agg = action['aggfunc']
    # decide if it's a valid place to insert
    if len(node_list) > 0:
        toinsert = random.choice(node_list)  
        #construct the expression to be inserted 
        left = sql.Column('UserId')
        op = tokens.Op('=')
        right = tokens.Literal(1)
        addBooleanCompare = logical.BooleanCompare(left, op, right)
        then = tokens.Literal(100000)
        when_expr = [logical.WhenExpression(addBooleanCompare,then)]
        else_expr = sql.Literal(1)
        addexpression = logical.CaseExpression(None, when_expr, else_expr)
        expr_tbi = sql.AggFunction(agg, None, addexpression)
        if hasattr(toinsert, 'parent') and isinstance(toinsert.parent, NamedExpression):
            toinsert.parent.expression = expr_tbi
        elif not hasattr(toinsert, 'parent'):
            expr_tbi = NamedExpression(None, expr_tbi)
            ast.select.namedExpressions.seq.append(expr_tbi)
    return ast

