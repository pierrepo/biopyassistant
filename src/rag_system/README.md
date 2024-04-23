# RAG SYSTEM 🤖

This repository contains the code for the Retrieval-Augmented Generation (RAG) system, which is a conversational AI model that combines the strengths of retrieval-based and generative models. The RAG model is fine-tuned on a dataset of Python programming courses and exercises to provide accurate and informative responses to user queries.


## Description 📄

The RAG system is built using the Langchain library, which offers flexibility in integrating various open-source models and tools. Leveraging Langchain allows for seamless integration of different models, including those from OpenAI, enhancing the system's adaptability and scalability.


The process of querying the chatbot involves several steps: First, the query text and customization options are retrieved from the command-line arguments. Next, the system loads the vector database containing relevant information. Then, it searches for documents in the database that are similar to the user query. Subsequently, the metadata of the top matching documents is extracted. Following this, a prompt is generated for the AI model based on the query and user preferences. The AI model predicts the response based on the prompt. If metadata inclusion is requested, it adds metadata to the response before displaying the results. Otherwise, it directly displays the response without metadata. This streamlined process ensures efficient and accurate interactions with the chatbot.


## 📝 Usage

To process the data, ensure you have activated the Conda environment, processed the data, and created the database as described in the main [README](../../README.md). Then, run the following command:

```bash
python src/rag_system/query_chatbot.py "Your question here" [--model "model_name"]
                                                            [--question-type "type"]
                                                            [--python-level "level"] 
                                                            [--include-metadata]
```

#### Customization options:

- 🤖 Model Selection: Choose between GPT-3 Turbo or GPT-4 to suit your needs and preferences. Default: "gpt-3-turbo".

- 🔍 Question Type: Specify the type of question. Options: "course" or "exercise". Default: "course".

- 📊 Python Level: Specify your proficiency level in Python. Options: "beginner", "intermediate", or "advanced". Default: "intermediate".

- 📝 Include Metadata: Include metadata in the response, such as the sources of the answer. By default, metadata is excluded.

> Remark: Currently, we are utilizing models provided by OpenAI.

