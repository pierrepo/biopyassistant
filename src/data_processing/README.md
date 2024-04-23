# Data processing 🔧

This directory contains the code for processing the data. The data is stored in the `data` directory. The data processing pipeline consists of the following steps:

1. **Cleaning python comments** 🧹: Remove spaces between '#' and comments in Python code blocks in Markdown content. This is done to avoid the comments being treated as headers in the markdown content.

2. **Renumbers the headers** 🔢: Renumber the headers in the markdown content. The headers are numbered continuously throughout the parsing of the files in the directory. Chapters and appendices are treated differently; chapters are numbered in numerical order, while appendices are numbered alphabetically. This renumbering process is aimed at making the addition of metadata clearer for students, as it is easier to navigate and reference numerical headers compared to chapter or section names.

## 📝 Usage

To process the data, ensure you have activated the Conda environment as described in the main [README](../../README.md). Then, run the following command:

```bash
    python src/data_processing/markdown_parser.py  [source_dir] [dest_dir]
```

where `source_dir` is the directory containing the markdown files to be processed and `dest_dir` is the directory where the processed files will be saved. By default, the source directory is `data/raw_python_courses` and the destination directory is `data/processed_python_courses`.


May your data tasks be a breeze! 💨