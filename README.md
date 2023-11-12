# Document Question and Answering using LangChain and Llama with web interface using Streamlit

LangChain is a framework for building conversational AI systems. It provides a modular and extensible architecture that allows developers to easily integrate different components, such as language models, knowledge bases, and dialogue managers.

Llama is a large language model (LLM) that can be used to generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way. It is trained on a massive dataset of text and code, and it can be used to power a variety of conversational AI applications.

**Together, LangChain and Llama provide a powerful platform for building conversational AI systems that are both informative and engaging.**

Streamlit is an open-source Python library that makes it easy to build and share web apps for machine learning and data science. It is designed to be easy to use, even for people with no experience in web development. With Streamlit, you can quickly create interactive web apps that display data, run machine learning models, and collect user input.

## Project Structure

```
.
├── app.py
├── data
│   └── dc.txt
├── models
│   └── llama-7b.ggmlv3.q4_0.bin
├── notebooks
│   └── notebook.ipynb
├── requirements.txt
└── venv
    ├── bin
    ├── etc
    ├── include
    ├── lib
    ├── lib64 -> lib
    ├── pyvenv.cfg
    └── share
```

## Description

In this repository I have created a question answering conversational AI system based on the uploaded text file. To try this out yourself, you can simply follow these steps:

1. Clone the repo
2. Create a virtual env ```python -m venv venv```
3. Install the dependencies ```pip install -r requirements.txt```
4. Run the streamlit app ```streamlit run app.py```