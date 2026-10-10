# Blinded image audit

Judge each image on its own. For every row in `label_sheet.csv`, open
the image named in the `image` column and answer the question in
`question` by writing `yes`, `no` or `unsure` in
`answer_yes_no_unsure`. Leave `notes` for anything ambiguous.

You are not told which model produced an image, which condition it
belongs to, or what an automatic detector thought. That is deliberate:
your labels are the reference these detector numbers are checked
against, so they must be formed independently.

Do not look for the key. It is stored outside this directory.
