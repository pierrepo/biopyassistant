
# BioPyAssistant

![Banner](/assets/img/banner.png)

*AI-powered conversational agent designed to help
biology students learn the Python programming language.*

## Introduction

This conversationnal agent (chatbot) is designed to help biology students learn
the Python programming language.It is based on the OpenAI models and provides
 answers to questions related to Python programming.

The chatbot uses the Retrieval-Augmented Generation (RAG) methodology to build its
responses from this [Python course](https://python.sdv.u-paris.fr/)
(Markdown files available in the
[Python course repository](https://github.com/bioinfo-prog/cours-python)).

## Setup

To install BioPyAssistant and its dependencies,
you need to perform the following steps:

### Clone the repository

```bash
git clone https://github.com/pierrepo/biopyassistant.git
cd biopyassistant
```

### Activate the environment

We use [uv](https://docs.astral.sh/uv/getting-started/installation/)
to manage dependencies and the project environment.

Sync dependencies:

```bash
uv sync
```

### Copy the raw Markdown files of the Python [course](https://github.com/bioinfo-prog/cours-python)

```bash
git clone --depth 1 https://github.com/bioinfo-prog/cours-python.git
rm -f data/course_raw/*.md
cp cours-python/cours/*.md data/course_raw/
rm -rf course-python
```

### Process raw Markdown files

```bash
rm -f data/course_processed/*.md
uv run parse-clean-markdown --config data/chapters_and_levels.yaml
```

In this step, Python comments (`#`) are slightly changed
to avoid confusion with Markdown headers (`#`, `##`...) and
headers are numbered (from `## Title` to `## 1.1 Title`).
Processed Markdown files are stored in `data/course_processed`

### Add OpenAI and OpenRouter API key

Create an .env file with a valid [OpenAI](https://platform.openai.com/docs/api-reference/authentication)
and [OpenRouter](https://openrouter.ai/docs/api/reference/authentication) API key:

```sh
OPENAI_API_KEY=<your-openai-api-key>
OPENROUTER_API_KEY=<your-openrouter-api-key>
```

> Remark: This .env file is ignored by git.

### Create the vector database

```bash
uv run create-database --course-yaml data/chapters_and_levels.yaml \
                       --chroma-path vectorstores/chroma_db \
                       --embedding-model text-embedding-3-large \
                       --model-provider openai \
                       --chunk-size 1000 --chunk-overlap 200
```

This command will create a Chroma vector database from the processed Markdown files.
All files will be split into chunks of 1000 characters with an overlap of 200 characters.

> Remark: The vector database is saved on disk.

## Usage (command line interface)

```bash
uv run query-chatbot  --query "Your question here" \
                      --level "user_level" \
                      --model "model_name" \
                      --provider-llm "provider_name" \
                      --include-metadata
```

### Options

- 📚 **User Level**: Specify the user's Python knowledge level to tailor
                     the chatbot's responses.
                     Choose between: `beginner`, `intermediate`, `advanced`.
- 🤖 **Model Selection**: Choose the language model for the query.
                          Examples: `gpt-4o`, `deepseek/deepseek-v3.2`, etc.
- 🌐 **LLM Provider**: Specify the provider of the language model.
                       Choose between: `openai`, `openrouter`.
- 📝 **Include Metadata**: Include metadata in the response,
                          such as the sources of the answer.
                          By default, metadata is excluded.

Example:

```bash
uv run query-chatbot  --query "What is the difference between list and set ?" \
                      --level "advanced" \
                      --model "gpt-4o" \
                      --provider-llm "openai" \
                      --include-metadata
```

This command will query the chatbot for a response to the question
"What is the difference between list and set ?"
for an intermediate user using the `gpt-4o` model from the `openai` provider.
The response will include metadata about the sources of the answer.

Output:

```text
Query:
What is the difference between list and set ?

Response:
A list is an ordered collection of elements, while a set is an unordered collection
of unique elements.In a list, the order of elements is preserved, and duplicate
elements are allowed. In contrast, a set does not preserve the order of elements,
and duplicate elements are not allowed. Additionally, a set is optimized for
membership testing and eliminating duplicate elements,
making it more efficient for certain operations than a list.

For more information, you can refer to the following sources:
- Chapter ... (Link to the source : ...)
- Chapter ... (Link to the source : ...)
```

## Usage (web interface)

```bash
uv run streamlit run src/biopyassistant/ui/streamlit_app.py
```

This will run the Streamlit app in your web browser.
