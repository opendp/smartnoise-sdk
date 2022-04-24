ANTLR_LIB := /usr/local/lib/antlr-4.9.3-complete.jar
ANTLR := java -jar $(ANTLR_LIB)
GRUN := java -classpath ".:$(ANTLR_LIB)" org.antlr.v4.gui.TestRig
PYTHON := python3

SqlSmallVisitor.py: SqlSmall.g4
	$(ANTLR) -visitor -Dlanguage=Python3 SqlSmall.g4

SqlSmallParser.java: SqlSmall.g4
	$(ANTLR) -visitor SqlSmall.g4

SqlSmallParser.class: SqlSmallParser.java
	javac *.java

gui: SqlSmallParser.class
	$(GRUN) SqlSmall batch -tokens -gui -diagnostics

clean:
	rm SqlSmallListener* || true
	rm SqlSmallVisitor* || true
	rm SqlSmallLexer* || true
	rm SqlSmallParser* || true
	rm SqlSmallBase* || true
	rm *.interp || true
	rm *.tokens || true
	rm *.class || true
	rm *.java || true