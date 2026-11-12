# Raw datasets

These directories contain the source files unpacked from the dataset archives supplied for this project. The preprocessing pipeline reads these local files; it does not download datasets at run time.

`default_credit_card_clients/` contains the original `.xls` plus an `.xlsx` conversion of the same workbook. The reader prefers the `.xlsx` copy (OpenPyXL) so that installation does not depend on the legacy XLS parser. The original file is retained for provenance.

Do not overwrite raw files. Generated, cleaned matrices belong under `data/processed/`.
