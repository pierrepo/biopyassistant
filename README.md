
<h1 align="center">
  <img style="vertical-align:middle; width:50%; position:fixed;"
  src="/data/img/banner.png">
</h1>
<p align="center">
  <i>AI-powered conversational agent designed to assist biology students</i><br>
  <i>in learning the Python programming language.</i>
</p>

<p align="center">
    <a href="https://github.com/pierrepo/biopyassistant/releases">
        <img alt="Website" src="https://img.shields.io/website?url=https%3A%2F%2Fgithub.com%2Fpierrepo%2Fbiopyassistant&up_message=click%20here%20!&color=%23539fc9">
    </a>
      <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=%23539fc9">
    </a>
    <a href="https://github.com/pierrepo/biopyassistant/blob/main/LICENSE">
        <img alt="GitHub License" src="https://img.shields.io/github/license/pierrepo/biopyassistant?style=flat&color=%23539fc9&link=https%3A%2F%2Fgithub.com%2Fpierrepo%2Fbiopyassistant%2Fblob%2Fmain%2FLICENSE">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#-installation">Installation</a> |
        <a href="#-usage">Usage</a>
    </p>
</h4>


## Installation

To install BioPyAssistant and its dependencies, run the following commands:

Clone the repository:

```bash
git clone https://github.com/pierrepo/biopyassistant.git
cd biopyassistant
```

Install Conda:

To install Conda, follow the instructions provided in the official [Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Create a Conda environment:

```bash
conda env create -f environment.yml
```


## Usage

### Step 1: Activate the Conda Environment

Activate the Conda environment by running:

```bash
conda activate biopyassistantenv
```

### Step 2: Process the course content

Process the course content by running:

```bash
python src/parse_clean_markdown.py --in data/markdown_raw --out data/markdown_processed
```

This command will process Markdown files located in the `data/markdown_raw` directory and save the processed files to the `data/markdown_processed` directory.

### Step 3: Set up OpenAI API key

Create a `.env` file with a valid OpenAI API key:

```text
OPENAI_API_KEY=<your-openai-api-key>
```

> Remark: This `.env` file is ignored by git.

### Step 4: Create the Vector Database

Create the Vector database by running:

```bash
python src/create_database.py --data-path [data-path] --chroma-path [chroma-path] --chunk-size [chunk-size] --chunk-overlap [chunk-overlap] 
```
Where :
- `[data-path]` (mandatory): Directory containing processed Markdown files.
- `[chroma-path]` (mandatory): Output path to save the vectorial ChromaDB database.
- `[chunk-size]` (optional): Size of text chunks to create. Default: 1000.
- `[chunk-overlap]` (optional): Overlap between text chunks. Default: 200.

Example:
  
```bash
python src/create_database.py --data-path data/markdown_processed --chroma-path chroma_db
```
This command will create a vectorial Chroma database from the processed Markdown files located in the `data/markdown_processed` directory. The text will be split into chunks of 1000 characters with an overlap of 200 characters. And finally the vectorial Chroma database will be saved to the `chroma_db` directory.

> Remark: The vector database will be saved on the disk.


### Step 5: Query the chatbot.

You can query the chatbot using either the command line or the graphical interface:


#### **Command Line**

```bash
python src/query_chatbot.py --query "Your question here" [--model "model_name"]  [--include-metadata]
```

#### Customization options:

- 🤖 Model Selection: Choose between `gpt-4o`, `gpt-4-turbo`, `gpt-4` and `gpt-3.5-turbo` to suit your needs and preferences. Default: `gpt-3.5-turbo`.

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


#### **Graphical Interface** :

##### Streamlit App:

Run the following command:

```bash
streamlit run src/streamlit_app.py
```

This will launch the Streamlit app in your browser, where you can start interacting with the RAG model.

##### Gradio App:

Run the following command:

```bash
python src/gradio_app.py
```

This will launch the Gradio app in your browser, where you can start interacting with the RAG model.

