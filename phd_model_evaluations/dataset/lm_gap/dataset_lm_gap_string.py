HEADER_IN_TEXT = """dataset_name\tleft_context\tright_context
"""

HEADER_OUT_TEXT = """gap_token
"""

CONFIG_TEXT = """--metric PerplexityHashed --precision 6 --in-header in-header.tsv --out-header out-header.tsv
"""

README_TEXT = """# LM-GAP CHALLENGE

LM-GAP challenge is the task to **predict token between left and right context of text**.
Sometimes, left or right context may not exist.
Predicted token, left or right context is not tokenized, which means predicted token may be a token with punctuation.

# Dataset structure

There are 3 datasets:

- train set in `train` directory,
- validation set in `dev-0` directory,
- test set in `test-A` directory - this dataset is without expected tokens.

## Structure of each dataset

- `in.tsv` - input TSV file (`\\t` - tabulation is column separator), where: first column is the dataset name,
second column is the left context, third column is the right context - name of column is available
in `in-header.tsv` file, each line is a separate example,
- `expected.tsv` - expected token in the gap (for test set is not available),
this is one column file - name of column is available in `out-header.tsv` file, each line is a separate example,
- `raw_data.txt` - raw text, each line is a separate line of text,
- `dataset_data.json` - datasets json data (maybe used with Python `datasets` library) - it's list of
dictionaries with keys:
  - `dataset_name` key with dictionary with dataset name, (optionally) dataset configuration and dataset split name,
  - `metric` key with metric configuration from Python `evaluate` library,
  - `feature_definitions` key with definition of type data columns from Python `datasets` library,
  - `data` key with dataset from Python `datasets` library.

## Structure of output file

Output file should be saved in file with `out` prefix (in dataset directory).
Lines in the output file should contain predicted tokens with probability separated by the space character.
Token and probability should be separated by ":" (colon) character.
Additionally, for each line should be added special probability for rest/unknown of tokens
(saved as `:PROBABILITY`, where `PROBABILITY` is the probability of rest/unknown of tokens) - without this token,
finally score can go to infinity (if predicted tokens are incorrect).

Example of output file is:
```shell
$ cat dev-0/out.tsv
skills:0.78 qualifications:0.12 qualities:0.09 :0.01
replace:0.52 substitute:0.32 surpass:0.15 :0.01
```

Example contains 2 examples with 3 best tokens (with probability for rest/unknown of tokens).

## Many output files

It's possible to save many output files in dataset directory.
To do that, output file should be saved as `out-TAGS.tsv` where `TAGS`
is key-value pairs separated with `,` (comma) character.
Each key-value pair should be separated `=` (sign equals).

Example `out-model=roberta-base,method=MLM.tsv` contains 2 tags: `model` with `roberta-base` value
and `method` with `MLM` value. Value/Key of tag should not contain `,` (comma) character.

# Evaluation

Evaluation can be done with [geval](https://gitlab.com/filipg/geval) tools.

Example of `dev-0` evaluation can be run:

```shell
geval -t dev-0
```

"""

GIT_IGNORE_TEXT = """*~
"""
