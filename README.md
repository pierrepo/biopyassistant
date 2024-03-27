
<div style="display: flex; align-items: left;">
  <img src="data/logo.webp" alt="Logo" width="40" height="40">
  <h1 align="center">BioPyAssist</h1>
</div>

BioPyAssist is a Python-based chatbot designed to assist users with questions related to Python, with a focus on applications in the field of biology.


## Installation

To install BioPyAssist and its dependencies, run the following command:

1. Clone the repository:

```bash
git clone https://github.com/pierrepo/biopyassist
cd biopyassist
```

2. Create and activate a new virtual environment (recommended):

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

3. Install the required dependencies:

```bash
conda env create -f environment.yml
```


## Usage

### Step 1: Install Requirements

First, install the required Python packages by running:

```bash
conda activate biopyassistenv
```

### Step 2: Create Chroma DB

Next, create the Chroma database by running:

```python
python src/create_database.py
```

### Step 3: Set up OpenAI Account

To use the OpenAI functionality, you need to set up an OpenAI account and obtain an API key. Once you have the key, set it in your environment variable as OPENAI_API_KEY.

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```