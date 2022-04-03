ANTLR_LIB := /usr/local/lib/antlr-4.9.3-complete.jar
ANTLR := java -jar $(ANTLR_LIB)
GRUN := java -classpath ".:$(ANTLR_LIB)" org.antlr.v4.gui.TestRig
PYTHON := python3

XPathVisitor.py: XPath.g4
	$(ANTLR) -visitor -Dlanguage=Python3 XPath.g4

XPathParser.java: XPath.g4
	$(ANTLR) -visitor XPath.g4

XPathParser.class: XPathParser.java
	javac *.java

gui: XPathParser.class
	$(GRUN) XPath statement -tokens -gui -diagnostics

clean:
	rm XPathListener* || true
	rm XPathVisitor* || true
	rm XPathLexer* || true
	rm XPathParser* || true
	rm XPathBase* || true
	rm *.interp || true
	rm *.tokens || true
	rm *.class || true
	rm *.java || true
	rm -rf .antlr || true