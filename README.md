
<div style="display: flex; align-items: left;">
  <img src="data/logo.webp" alt="Logo" width="40" height="40">
</div>

# BioPyAssistant

BioPyAssistant is a chatbot designed to assist students from life science curricula with questions related to Python programming.


## Installation

To install BioPyAssistant and its dependencies, run the following commands:

Clone the repository:

```bash
git clone https://github.com/pierrepo/biopyassistant.git
cd biopyassistant
```

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

### Step 2: Process the Data

Process the data by running:

```bash
python src/markdown_parser.py --in data/markdown_raw --out data/markdown_processed
```

This command will process Markdown files located in the `data/markdown_raw` directory and save the processed files to the `data/markdown_processed` directory.

### Step 3: Set up OpenAI API key

Create a `.env` file with a valid OpenAI API key:

```text
OPENAI_API_KEY=<your-openai-api-key>
```

> Remark: The `.env` file is ignored by git.

### Step 4: Create Chroma DB

Create the Chroma database by running:

```bash
python src/create_database.py [data_dir] [chunk_size] [chunk_overlap] [txt_output] [chroma_output]
```
Where :
- `[data_dir]` (optional): Directory containing processed Markdown files. Default: `data/markdown_processed`.
- `[chunk_size]` (optional): Size of text chunks to create. Default: 300.
- `[chunk_overlap]` (optional): Overlap between text chunks. Default: 100.
- `[txt_output]` (optional): Output file to save text chunks with metadata. Default: None.
- `[chroma_output]` (optional): Output path to save ChromaDB database. Default: `chroma_db`.


> Remark: The vector database will be saved on the disk.

### Step 5: Query the chatbot.

You can query the chatbot using either the command line or the graphical interface:


#### **Command Line** :

Run the following command:

```bash
python src/query_chatbot.py "Your question here" [--model "model_name"]
                                                  [--question-type "type"]
                                                  [--python-level "level"] 
                                                  [--include-metadata]
```

#### Customization options:

- 🤖 Model Selection: Choose between GPT-3 Turbo or GPT-4 to suit your needs and preferences. Default: "gpt-3-turbo".

- 🔍 Question Type: Specify the type of question. Options: "course" or "exercise". Default: "course".

- 📊 Python Level: Specify your proficiency level in Python. Options: "beginner", "intermediate", or "advanced". Default: "intermediate".

- 📝 Include Metadata: Include metadata in the response, such as the sources of the answer. By default, metadata is excluded.

#### **Graphical Interface** :

Run the following command:

```bash
streamlit run src/streamlit_app.py
```

This will launch the Streamlit app in your browser, where you can start interacting with the RAG model.