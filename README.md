# CAN
Collaborative Adversarial Networks 


## Topological Knowledge Graph for LLM Confidence Assessment

### Overview

This library provides a framework for creating a topological knowledge graph to assess and improve the confidence of Large Language Model (LLM) responses. It uses a smaller LLM (like Phi-3) with Retrieval-Augmented Generation (RAG) connected to a search engine to build a knowledge graph, and then grades a larger LLM's knowledge to model its understanding and predict potential hallucinations.

### Features

- Create a dynamic knowledge graph based on topics and their relationships
- Utilize a smaller LLM with RAG for building the knowledge graph
- Grade a larger LLM's knowledge on specific topics
- Predict hallucination risks in LLM responses
- Provide confidence scores and hallucination risk assessments for queries

### Requirements

- Python 3.7+
- NetworkX
- PyTorch
- Transformers
- Requests

### Installation

1. Clone this repository:
