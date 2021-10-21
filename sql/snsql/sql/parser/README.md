# Extending the Grammar

* Add grammar to .g4
    * Names at bottom
    * Resolution works upward
* Add parse-only tests
* Create AST objects
    * Make sure string round-trip works (use self.children)
    * Properly handle if certain syntax is optional
    * implement `evaluate` and `symbol`
* Add ast round-trip tests
* Add parse that goes to AST
* Add evaluation tests
* Add validate test
* Add rewrite tests
* Add tests to run  queries
