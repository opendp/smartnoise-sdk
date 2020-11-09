ANTLR := java -jar /usr/local/lib/antlr-4.8-complete.jar
GRUN := java org.antlr.v4.gui.TestRig
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