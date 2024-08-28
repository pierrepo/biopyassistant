
<h1 align="center">
  <img style="vertical-align:middle; width:70%; position:fixed;"
  src="/data/img/banner.png">
</h1>

<p align="center" style="width: 500px;">
  <i>AI-powered conversational agent designed to help biology students learn the Python programming language.
  </i>
</p>

<p align="center">
    <img alt="Made with Python" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=%23539fc9">
    <img alt="BSD-3 Clause License" src="https://img.shields.io/github/license/pierrepo/biopyassistant?style=flat&color=%23539fc9&link=https%3A%2F%2Fgithub.com%2Fpierrepo%2Fbiopyassistant%2Fblob%2Fmain%2FLICENSE">
</p>

## Introduction

This conversationnal agent (chatbot) is designed to help biology students learn the Python programming language. It is based on the OpenAI models and provides answers to questions related to Python programming.

The chatbot uses the Retrieval-Augmented Generation (RAG) methodology to build its responses from this [Python course](https://python.sdv.u-paris.fr/) (Markdown files available [here](https://github.com/bioinfo-prog/cours-python)).


## Setup

To install BioPyAssistant and its dependencies, you need to perform the following steps:

### Clone the repository

```bash
git clone https://github.com/pierrepo/biopyassistant.git
cd biopyassistant
```

### Install [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Create a Conda environment

```bash
conda env create -f environment.yml
```

### Activate the Conda environment

```bash
conda activate biopyassistantenv
```

### Copy the raw Markdown files of the Python [course](https://github.com/bioinfo-prog/cours-python):

```bash
git clone https://github.com/bioinfo-prog/cours-python.git
cp cours-python/cours/*.md data/markdown_raw/
rm -rf cours-python
```

### Process raw Markdown files

```bash
python src/parse_clean_markdown.py --in data/markdown_raw --out data/markdown_processed
```

In this step, Python comments (`#`) are slighty changed to avoid confusion with Markdown headers (`#`, `##`...) and headers are numbered (from `## Title` to `## 1.1 Title`). Processed Markdown files are stored in `data/markdown_processed`


### Add OpenAI API key

Create an `.env` file with a valid OpenAI API key:

```text
OPENAI_API_KEY=<your-openai-api-key>
```

> Remark: This `.env` file is ignored by git.


### Create the vector database

```bash
python src/create_database.py --data-path data/markdown_processed --chroma-path chroma_db
```

This command will create a Chroma vector database from the processed Markdown files. All files will be split into chunks of 1000 characters with an overlap of 200 characters. 

> Remark: The vector database is saved on disk.


## Usage (command line interface)


```bash
python src/query_chatbot.py --query "Your question here" [--model "model_name"]  [--include-metadata]
```

### Options

- 🤖 Model Selection: Choose between `gpt-4o`, `gpt-4-turbo`, `gpt-4` and `gpt-3.5-turbo`. Default: `gpt-3.5-turbo`.
- 📝 Include Metadata: Include metadata in the response, such as the sources of the answer. By default, metadata is excluded.

Example:

```bash
python src/query_chatbot.py --query "What is the difference between list and set ?" --model gpt-4-turbo --include-metadata
```

This command will query the chatbot with the question "What is the difference between list and set ?" using the `gpt-4-turbo` model and include metadata in the response.

Output:

```text
Query:
What is the difference between list and set ?

Response:
A list is an ordered collection of elements, while a set is an unordered collection of unique elements. In a list, the order of elements is preserved, and duplicate elements are allowed. In contrast, a set does not preserve the order of elements, and duplicate elements are not allowed. Additionally, a set is optimized for membership testing and eliminating duplicate elements, making it more efficient for certain operations than a list.

For more information, you can refer to the following sources:
- Chapter ... (Link to the source : ...)
- Chapter ... (Link to the source : ...)
```


## Usage (web interface)

### Streamlit app


```bash
streamlit run src/streamlit_app.py
```

This will run the Streamlit app in your web browser.


### Gradio App:


```bash
python src/gradio_app.py
```

This will run the Gradio app in your web browser.

