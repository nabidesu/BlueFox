# Retrieval-Augmented Generation (RAG) Chatbot

## Overview

This project explores the design and implementation of a Retrieval-Augmented Generation (RAG) chatbot. It combines retrieval from a document store with generative responses from a language model.

## Contents

- Preprocessing knowledge base documents
- Retrieval setup using vector similarity
- RAG pipeline integration
- Experiments with retrieval and generation components

## Features

- Custom chatbot capable of answering based on internal documents
- Modular code for retriever and generator
- Evaluation of chatbot performance with test queries

## Prerequisites

- Python 3.8+
- A text corpus for knowledge retrieval
- Required Python libraries:
  - transformers
  - sentence-transformers
  - faiss-cpu or faiss-gpu
  - flask (if web interface used)
