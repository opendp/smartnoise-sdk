# Automated Query Tests

Tests are run in increasing order of capabilities: parse, validate, rewrite, and then run.  Any set of queries named *_fail is designed to fail that stage.

## Parse

Simply tests the lexer, parser, and AST builder.  Ensures round-trip of text to AST and back to text.

## Validate

Tests the validator after pasting.  All queries, including *_fail under this folder, should successfully parse.  The *_fail queries here will fail validation.

## Rewrite 

Tests the rewriter.  Queries must be valid and parsed before passing through the rewriter.  These queries are used in parse and validate tests as well.

## Run

These queries need to reference a database that has the appropriate test tables installed.  These queries are run against the database, with actual and noisy answers compared.  These queries must be valid, so they are also used in parse, validate, and rewrite testing.