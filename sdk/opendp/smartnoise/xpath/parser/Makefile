ANTLR := java -jar /usr/local/lib/antlr-4.8-complete.jar
GRUN := java org.antlr.v4.gui.TestRig
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