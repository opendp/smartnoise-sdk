# Auto Generate Rest Client and Models with AutoRest


**Warnings: DO NOT manually edit auto-generated files**.

## Getting Started

To build the SDKs for Rest APIs, simply install AutoRest via `npm` (`npm install -g autorest`), move command prompt to ./openapi folder, and run:
> `autorest readme.md`

To see additional help and options, run:
> `autorest --help

For other options on installation see [Installing AutoRest](https://aka.ms/autorest/install) on the AutoRest github page.

## Edit Swagger Specification File for Update
- These input files are listed in swagger folder. Edit these files for updating client definition.
- You can learn [how to write this specification](https://github.com/Azure/autorest/blob/master/docs/developer/guide/defining-clients-swagger.md)

## Inputs
We want a single client to be created from the following OpenAPI definition files:

``` yaml
input-file:
  - swagger.yml
```
```yaml
title: rest_client
```

## Generation
```yaml
python:
    add-credentials: true
    namespace: restclient
    no-namespace-folders: true
    output-folder: restclient
    package-version: 1.5.0
    base-folder: ./../sdk/burdock/
```

## Temporary output folder to save output-artifact and source maps files. Please do not check in these files.

``` yaml
output-folder: swagger_generation_log
```



### Fully resolved OpenAPI definition

To support tools unable to process multiple OpenAPI definitions or definitions with external references (`$ref: "<URI to another OpenAPI definition>#/definitions/SomeModel"`), AutoRest allows exporting a single, fully resolved OpenAPI definition without any external references that tools should be able to consume.

``` yaml
output-artifact:
  - swagger-document.norm.json
  - swagger-document.norm.yaml
```

### Source maps

AutoRest tries to create source maps for output artifacts. These will relate the artifact with the original input files which may be helpful for tools created around AutoRest.
For example, AutoRest uses the source map internally in order to relate validation messages back to the original files.

``` yaml
output-artifact:
  - swagger-document.norm.json.map
  - swagger-document.norm.yaml.map
```

