grammar XPath;

statement
    : innerStatement
    EOF
    ;

innerStatement
    : (childSelector | rootSelector | rootDescendantSelector) (indexSelector | ('/' childSelector) | '//' (descendantSelector))*
    ;

childSelector
    : (ident=IDENTIFIER | attr=ATTRIBUTE | allsel=allSelect) booleanSelector?
    ;

rootSelector
    : '/' (ident=IDENTIFIER | attr=ATTRIBUTE | allsel=allSelect) booleanSelector?
    ;

rootDescendantSelector
    : '//' (ident=IDENTIFIER | attr=ATTRIBUTE | allsel=allSelect) booleanSelector?
    ;

descendantSelector
    : (ident=IDENTIFIER | attr=ATTRIBUTE | allsel=allSelect) booleanSelector?
    ;

booleanSelector
    : '[' (left=innerStatement) (op=comparisonOperator (rlit=literal | stmt=innerStatement))? ']'
    | '[' llit=literal op=comparisonOperator (rlit=literal | stmt=innerStatement) ']'
    ; 

indexSelector
    : '[' index=INTEGER_VALUE ']'
    ;

allSelect
    : allAttributes
    | allNodes
    ;

allNodes
    : '*'
    ;

allAttributes
    : '@*'
    ;

comparisonOperator
    : EQ | NEQ | NEQJ | LT | LTE | GT | GTE 
    ;

booleanValue
    : TRUE | FALSE
    ;

allExpression
    : ASTERISK
    ;

literal 
    : STRING    #stringLiteral 
    | number    #numberLiteral 
    | TRUE      #trueLiteral
    | FALSE     #falseLiteral 
    | NULL      #nullLiteral
    ;


number
    : MINUS? DECIMAL_VALUE            #decimalLiteral
    | MINUS? INTEGER_VALUE            #integerLiteral
    ;

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

TRUE: T R U E;
FALSE: F A L S E;
NULL: N U L L;

/*
    Standard Lexer stuff
*/
STRING
    : '\'' ( ~('\''|'\\') | ('\\' .) )* '\''
    ;

INTEGER_VALUE
    : DIGIT+
    ;

DECIMAL_VALUE
    : DIGIT+ EXPONENT
    | DECIMAL_DIGITS EXPONENT? 
    ;

ATTRIBUTE
    : '@' IDENTIFIER
    ;

IDENTIFIER
    : LETTER+ (LETTER | DIGIT | '_')*
    ;

LETTER : (UCASE | LCASE);

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

WS : [ \t\r\n]+ -> skip ;

SPACE : [ \t]+ -> skip ;