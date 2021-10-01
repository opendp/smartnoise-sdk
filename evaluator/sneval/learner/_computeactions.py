from opendp.smartnoise.evaluation.learner._transformation import *
from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise._ast import tokens


def compute_action(ep: LearnerParams):
    columns = ep.columns
    SUM = tokens.FuncName("SUM")
    COUNT = tokens.FuncName("COUNT")
    AVG = tokens.FuncName("AVG")

    OP = [tokens.Op("+"), tokens.Op("-"), tokens.Op("*"), tokens.Op("/"), tokens.Op("%")]
    COLUMNS = columns + ["Allcolumns"]
    AGGFUNC = [SUM, COUNT, AVG]
    MAXNODELEN = ep.MAXNODELEN
    METHOD = [
        # modify nodes
        AddNumericLiteral,
        SubstractNumericLiteral,
        ReplaceAggFunction,
        ReplaceColumn,
        ReplaceOP,
        # delete nodes
        DeleteNode,
        # insert nodes
        InsertArithmeticExpression,
        InsertCaseExpression,
        InsertNamedExpression,
    ]

    actions = []
    expressions = {}

    for m in METHOD:
        for a in AGGFUNC:
            for c in COLUMNS:
                for o in OP:
                    action = {"method": m, "aggfunc": a, "column": c, "op": o, "ep": ep}
                    description = "{} using AggFunc: {}, Op: {} and Column: {}".format(
                        m.__name__, a, o, c
                    )
                    action["description"] = description
                    actions.append(action)
    return actions
