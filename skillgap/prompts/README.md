These templates ask for JSON only. The wrapper supplies the Pydantic-derived
JSON schema and wraps list/object output in an `items` field. Input documents
are serialized as data after the instructions. Source-quote validators and
grading citation checks reject invented quotes even when JSON is valid.

The templates are included in parser cache fingerprints. Change `prompt_version`
in config when changing extraction policies outside the template files.
