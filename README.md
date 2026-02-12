# Perceptrome CLI Quickstart

## Generate and use a custom catalog

You can generate a catalog from built-in accession category files and then pass it to `stream`.

```bash
perceptrome catalog-generate --categories plasmid bacteria --output custom_catalog.txt
perceptrome stream --catalog custom_catalog.txt --max-epochs 3
```

## Category semantics

`catalog-generate --categories` accepts these categories:

- `archaea`
- `bacteria`
- `chloroplast`
- `eukaryote`
- `metagenome`
- `mitochondrion`
- `plasmid`
- `synthetic_construct`
- `viroid`
- `viruses` (currently an alias of `viroid`, backed by `accessions/viroid_accessions.txt`)

If you need a dedicated virus accession source (not viroids), add a separate accession file and category mapping before using `viruses` for that purpose.
