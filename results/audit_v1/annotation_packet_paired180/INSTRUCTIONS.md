# Blinded image audit

Judge each image on its own. For every row in `label_sheet.csv`, open
the image named in the `image` column and answer the question in
`question` by writing `yes`, `no` or `unsure` in
`answer_yes_no_unsure`. Use `notes` for anything ambiguous.

Judge **only** the category named in `category_to_judge`. Whether the
image is beautiful, odd or damaged is not the question; record that in
`notes` instead.

You are not told which model produced an image, which condition it
belongs to, or what an automatic detector thought. That is deliberate:
your labels are the reference the detector numbers are checked
against, so they must be formed independently.

Some images will look similar to each other. Judge each one on its
own and do not try to group them.

Do not look for the key. It is stored outside this directory.
