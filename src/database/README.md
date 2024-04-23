# Database Repository 📁

This directory contains scripts for creating a vectorized chroma database from Markdown files. The create_database.py script generates the database from Markdown files stored in the specified directory (default: 'processed_data'). The database is used for retrieval and generation tasks in the RAG model.

## Description 📄

The create_database.py script creates the ChromaDB database from Markdown files in the specified directory. It first loads the Markdown documents, concatenates their content, and extracts the file names. Then, it splits the concatenated content into chunks based on headers and word limits. Next, it saves the text chunks to the ChromaDB using the text-embedding-3-large model from OpenAIEmbeddings. The script provides customization options for specifying the directory containing the Markdown files. By default, it processes Markdown files located in the processed_python_courses directory. This database creation process facilitates efficient retrieval and generation of responses in the RAG system, enhancing its performance and accuracy.



## 📝 Usage

To create the database, ensure you have activated the Conda environment and processed the data as described in the main [README](../../README.md).

> Remark: The vector database will be created in the 'chroma_db/' directory within the repository (on the disk).

Run the following command:

```bash
    python src/database/create_database.py [--data_dir]
```

> You can optionally specify the directory containing the Markdown files. If not specified, the default directory is processed_data. It contains the processed Markdown files from the 'data' directory.


Happy vectorizing ! 📚
