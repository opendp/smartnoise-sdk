grammar SqlSmall;

batch :
    query
    (';' query)*
    (';')?
    EOF
    ;

query :
      selectClause
      fromClause
      whereClause?
      aggregationClause?
      havingClause?
      orderClause?
      limitClause?
    ;

subquery :
    '('
    selectClause
    fromClause
    whereClause?
    aggregationClause?
    havingClause?
    orderClause?
    ')'
    ;

expressionSubquery : subquery;

selectClause
    : SELECT (setQuantifier)? namedExpressionSeq
    ;


fromClause : FROM relation (',' relation)*;

whereClause
    : WHERE booleanExpression
    ;

aggregationClause
    : GROUP BY groupingExpressions+=expression (',' groupingExpressions+=expression)*
    ;

havingClause
    : HAVING booleanExpression
    ;

orderClause
    : (ORDER BY order+=sortItem (',' order+=sortItem)*)
    ;

limitClause : LIMIT n=number;

topClause : TOP n=number;

joinRelation
    : (joinType) JOIN right=relationPrimary joinCriteria?
    ;

joinType
    : INNER?
    | CROSS
    | LEFT OUTER?
    | LEFT? SEMI
    | RIGHT OUTER?
    | FULL OUTER?
    | LEFT? ANTI
    ;

joinCriteria
    : ON booleanExpression                          #booleanJoin
    | USING '(' identifier (',' identifier)* ')'    #usingJoin
    ;

sortItem
    : expression ordering=(ASC | DESC)?
    ;

setQuantifier
    : DISTINCT
    | ALL
    | (topClause)
    ;

relation
    : relationPrimary joinRelation*
    ;

relationPrimary
    : qualifiedTableName (AS alias=identifier)?  #table
    | subquery (AS alias=identifier)?   #aliasedQuery
    | '(' relation ')' (AS alias=identifier)?     #aliasedRelation
    ;

caseExpression
    : CASE baseCaseExpr=expression (whenBaseExpression)+ (ELSE elseExpr=expression)? END #caseBaseExpr
    | CASE (whenExpression)+ (ELSE elseExpr=expression)? END #caseWhenExpr
    ;

namedExpression
    : expression (AS name=identifier)?
    ;

namedExpressionSeq
    : namedExpression (',' namedExpression)*
    ;

whenExpression : (WHEN baseBoolExpr=booleanExpression THEN thenExpr=expression);
whenBaseExpression : (WHEN baseWhenExpr=expression THEN thenExpr=expression);

expression
    : name=qualifiedColumnName                                                       #columnName
    | left=expression op=ASTERISK right=expression          #multiply
    | left=expression op=SLASH right=expression             #divide
    | left=expression op='%' right=expression               #modulo
    | left=expression op=PLUS right=expression              #add
    | left=expression op=MINUS right=expression             #subtract
    | caseExpression                                        #caseExpr
    | castExpression                                        #castExpr
    | allExpression                                         #allExpr
    | literal                                               #literalExpr
    | rankingFunction                                       #rankFunction
    | functionExpression                                    # functionExpr
    | expressionSubquery                                    #subqueryExpr
    | '(' expression ')'                                    #nestedExpr
    | stringFunction                                        #stringFunc
    | getTime                                               #currentTimeFunc
    | extractFunction                                       #extractFunc
    | dayNameFunction                                       #dayNameFunc
    ;


predicate
    : NOT? kind=BETWEEN lower=expression AND upper=expression #betweenCondition
    | NOT? kind=IN '(' expression (',' expression)* ')' #inCondition
    | IS NOT? kind=(NULL | TRUE | FALSE) #isCondition
    ;

functionExpression
    : bareFunction              #bareFunc
    | roundFunction             #roundFunc
    | powerFunction             #powerFunc
    | truncFunction             #truncFunc
    | function=aggregateFunctionName '(' setQuantifier? expression ')' #aggFunc
    | function=mathFunctionName '(' expression ')' #mathFunc
    | IIF '(' test=booleanExpression ',' yes=expression ',' no=expression ')' #iifFunc
    | CHOOSE '(' index=expression (',' literal)+ ')' # chooseFunc
    ;

booleanExpression
    : NOT booleanExpression   #logicalNot
    | left=expression op=comparisonOperator right=expression  #comparison
    | left=booleanExpression AND right=booleanExpression #conjunction
    | left=booleanExpression OR right=booleanExpression #disjunction
    | '(' booleanExpression ')' #nestedBoolean
    | expression predicate #predicated
    | name=qualifiedColumnName #boolColumn
    ;

castExpression: CAST '(' fromExpr=expression AS dbType ')';

dbType
    : INTEGER
    | FLOAT
    | BOOLEAN
    | TIMESTAMP
    | DATE
    | TIME
    | variableString
    | fixedString
    ;

variableString: (VARCHAR | NVARCHAR) ( '(' varCharLength=INTEGER_VALUE | MAX ')')?;
fixedString: (CHAR | NCHAR) ( '(' fixedCharLength=INTEGER_VALUE ')')?;

bareFunction : function=bareFunctionName '(' ')';

rankingFunction: function=rankingFunctionName  '(' ')' overClause;

comparisonOperator
    : EQ | NEQ | NEQJ | LT | LTE | GT | GTE | NSEQ
    ;

booleanValue
    : TRUE | FALSE
    ;

allExpression
    : ASTERISK
    | identifier '.' ASTERISK
    ;

literal
    : STRING    #stringLiteral
    | number    #numberLiteral
    | TRUE      #trueLiteral
    | FALSE     #falseLiteral
    | NULL      #nullLiteral
    ;

// date and time functions

extractFunction: EXTRACT '(' datePart FROM sourceExpr=expression ')';
dayNameFunction: DAYNAME '(' expr=expression ')';

datePart
    : YEAR
    | MONTH
    | DAY
    | HOUR
    | MINUTE
    | SECOND
    | MICROSECOND
    | WEEKDAY
    ;

getTime
    : CURRENT_DATE
    | CURRENT_TIME
    | CURRENT_TIMESTAMP
    ;

// string functions
stringFunction
    : stringUpper
    | stringLower
    | stringConcat
    | coalesceFunction
    | trimFunction
    | substringFunction
    | positionFunction
    | charLengthFunction
    ;

stringUpper: UPPER '(' sourceExpr=expression ')';
stringLower: LOWER '(' sourceExpr=expression ')';
stringConcat: CONCAT '(' expression (',' expression)* ')';
coalesceFunction: COALESCE '(' expression (',' expression)* ')';
trimFunction: TRIM '(' sourceExpr=expression ')';
substringFunction: SUBSTRING '(' sourceExpr=expression FROM startIdx=expression (FOR length=expression)? ')';
positionFunction: POSITION '(' searchString=expression IN sourceString=expression ')';
charLengthFunction: CHAR_LENGTH '(' sourceString=expression ')';

// numeric functions

truncFunction: (TRUNC | TRUNCATE) '(' sourceExpr=expression (',' digits=number)? ')';
roundFunction : ROUND '(' expression (',' digits=number)? ')';
powerFunction : POWER '(' expression ',' number ')';
mathFunctionName : ABS | CEIL | CEILING | FLOOR | SIGN | SQRT | SQUARE | EXP | LN | LOG | LOG10 | LOG2 | SIN | COS | TAN | ASIN | ACOS | ATAN | ATANH  | DEGREES;
bareFunctionName : PI | RANDOM | RAND | NEWID;


rankingFunctionName : ROW_NUMBER | RANK | DENSE_RANK;

aggregateFunctionName : COUNT | SUM | AVG | VAR | VARIANCE | STD | STDDEV | STDEV | MIN | MAX | PERCENTILE_DISC | PERCENTILE_CONT;

overClause : OVER '(' (PARTITION BY expression)? (orderClause)? ')';

aliasedSubquery : subquery (AS alias=identifier)?;

aliasedTableOrSubquerySeq : (aliasedTableName | aliasedSubquery) (',' (aliasedTableName | aliasedSubquery))*;

aliasedTableSeq : aliasedTableName (',' aliasedTableName)*;

aliasedTableName : qualifiedTableName (AS alias=identifier)?;

qualifiedTableName : QN3 | QN2 | IDENT;

qualifiedColumnName
    : QN2
    | IDENT
    | '"' STRING '"'
    ;

identifier: IDENT;

number
    : MINUS? DECIMAL_VALUE            #decimalLiteral
    | MINUS? INTEGER_VALUE            #integerLiteral
    ;

ABS: A B S;
ACOS: A C O S;
ALL: A L L;
AND: A N D;
ANTI: A N T I;
AS: A S;
ASC: A S C;
ASIN: A S I N;
ATAN: A T A N;
ATANH: A T A N H;
AVG: A V G;
BETWEEN: B E T W E E N;
BOOLEAN: B O O L E A N;
BY: B Y;
CASE: C A S E;
CAST: C A S T;
CHAR: C H A R;
CHAR_LENGTH: C H A R UNDERSCORE L E N G T H;
CEIL: C E I L;
CEILING: C E I L I N G;
CHOOSE: C H O O S E;
COALESCE: C O A L E S C E;
CONCAT: C O N C A T;
COS: C O S;
COT: C O T;
COUNT: C O U N T;
CROSS: C R O S S;
CURRENT_DATE: C U R R E N T UNDERSCORE D A T E;
CURRENT_TIME: C U R R E N T UNDERSCORE T I M E;
CURRENT_TIMESTAMP: C U R R E N T UNDERSCORE T I M E S T A M P;
DATE: D A T E;
DAY: D A Y;
DAYNAME: D A Y N A M E;
DEGREES: D E G R E E S;
DENSE_RANK: D E N S E '_' R A N K;
DESC: D E S C;
DISTINCT: D I S T I N C T;
DIV: D I V;
ELSE: E L S E;
END: E N D;
EXP: E X P;
EXTRACT: E X T R A C T;
FALSE: F A L S E;
FLOAT: F L O A T;
FLOOR: F L O O R;
FOR: F O R;
FROM: F R O M;
FULL: F U L L;
GROUP: G R O U P;
HAVING: H A V I N G;
HOUR: H O U R;
IF: I F;
IIF: I I F;
IN: I N;
INNER: I N N E R;
INTEGER: I N T E G E R;
INTERSECT: I N T E R S E C T;
IS: I S;
JOIN: J O I N;
LEFT: L E F T;
LIMIT: L I M I T;
LN: L N;
LOG: L O G;
LOG10: L O G '1' '0';
LOG2: L O G '2';
LOWER: L O W E R;
MAX: M A X;
MICROSECOND: M I C R O S E C O N D;
MIN: M I N;
MINUTE: M I N U T E;
MONTH: M O N T H;
NCHAR: N C H A R;
NEWID: N E W I D;
NOT: N O T;
NULL: N U L L;
NUMERIC: N U M E R I C;
ON: O N;
OR: O R;
ORDER: O R D E R;
OUTER: O U T E R;
OVER: O V E R;
PARTITION: P A R T I T I O N;
PERCENTILE_CONT: P E R C E N T I L E '_' C O N T;
PERCENTILE_DISC: P E R C E N T I L E '_' D I S C;
PI: P I;
POSITION: P O S I T I O N;
POWER: P O W E R;
RAND: R A N D;
RANDOM: R A N D O M;
RANK: R A N K;
RIGHT: R I G H T;
ROUND: R O U N D;
ROW_NUMBER: R O W '_' N U M B E R;
ROWNUM: R O W N U M;
SECOND: S E C O N D;
SELECT: S E L E C T;
SEMI: S E M I;
SIGN: S I G N;
SIN: S I N;
SORT: S O R T;
SQL: S Q L;  // reserved
SQRT: S Q R T;
SQUARE: S Q U A R E;
STD: S T D;
STDDEV: S T D D E V;
STDEV: S T D E V;
SUBSTRING: S U B S T R I N G;
SUM: S U M;
TAN: T A N;
THEN: T H E N;
TIME: T I M E;
TIMESTAMP: T I M E S T A M P;
TOP: T O P;
TRIM: T R I M;
TRUE: T R U E;
TRUNC: T R U N C;
TRUNCATE: T R U N C A T E;
TZOFFSET: T Z O F F S E T;
UNION: U N I O N; // reserved
UPPER: U P P E R;
USING: U S I N G;
VAR: V A R;
VARCHAR: V A R C H A R;
NVARCHAR: N V A R C H A R;
VARIANCE: V A R I A N C E;
WEEKDAY: W E E K D A Y;
WHEN: W H E N;
WHERE: W H E R E;
YEAR: Y E A R;

EQ  : '=' | '==';
NSEQ: '<=>';
NEQ : '<>';
NEQJ: '!=';
LT  : '<';
LTE : '<=' | '!>';
GT  : '>';
GTE : '>=' | '!<';

PLUS: '+';
MINUS: '-';
ASTERISK: '*';
SLASH: '/';
PERCENT: '%';
TILDE: '~';
AMPERSAND: '&';
PIPE: '|';
CONCAT_PIPE: '||';
HAT: '^';
UNDERSCORE: '_';


/*
    Standard Lexer stuff
*/
STRING
    : '\'' ( ~('\''|'\\') | ('\\' .) )* '\'';
//    | '"' ( ~('"'|'\\') | ('\\' .) )* '"'
//    ;


INTEGER_VALUE
    : DIGIT+
    ;

DECIMAL_VALUE
    : DIGIT+ EXPONENT
    | DECIMAL_DIGITS EXPONENT?
    ;


QN2 : IDENT '.' IDENT;

QN3 : IDENT '.' IDENT '.' IDENT;

IDENT: IDENTIFIER | ESCAPED_IDENTIFIER;

IDENTIFIER_UNICODE : [a-zA-Z_\u00A1-\uFFFF][a-zA-Z_\u00A1-\uFFFF0-9$]*;

IDENTIFIER
    : LETTER+ (LETTER | DIGIT | '_')*
    | DOUBLEQ_STRING_LITERAL
    | IDENTIFIER_UNICODE
    ;

ESCAPED_IDENTIFIER
    : '[' (LETTER | DIGIT | '_' | ' ' | '-')*? ']'
    | '"' (LETTER | DIGIT | '_' | ' ' | '-')*? '"'
    | '`' (LETTER | DIGIT | '_' | ' ' | '-')*? '`'
    ;

LETTER : (UCASE | LCASE);
DOUBLEQ_STRING_LITERAL : DQUOTA_STRING;

fragment DQUOTA_STRING : '"' ( '\\'. | '""' | ~('"' | '\\') )* '"';
fragment DECIMAL_DIGITS
    : DIGIT+ '.' DIGIT*
    ;

fragment EXPONENT
    : 'E' [+-]? DIGIT+
    ;

fragment DIGIT
    : [0-9]
    ;

fragment UCASE: [A-Z];
fragment LCASE: [a-z];

fragment A : [aA];
fragment B : [bB];
fragment C : [cC];
fragment D : [dD];
fragment E : [eE];
fragment F : [fF];
fragment G : [gG];
fragment H : [hH];
fragment I : [iI];
fragment J : [jJ];
fragment K : [kK];
fragment L : [lL];
fragment M : [mM];
fragment N : [nN];
fragment O : [oO];
fragment P : [pP];
fragment Q : [qQ];
fragment R : [rR];
fragment S : [sS];
fragment T : [tT];
fragment U : [uU];
fragment V : [vV];
fragment W : [wW];
fragment X : [xX];
fragment Y : [yY];
fragment Z : [zZ];

SIMPLE_COMMENT
    : '--' ~[\r\n]* '\r'? '\n'? -> skip
    ;

BRACKETED_EMPTY_COMMENT
    : '/**/' -> skip
    ;

BRACKETED_COMMENT
    : '/*' ~[+] .*? '*/' -> skip
    ;


WS : [ \t\r\n]+ -> skip ;

SPACE : [ \t]+ -> skip ;