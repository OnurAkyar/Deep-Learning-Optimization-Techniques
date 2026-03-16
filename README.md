# 🚀 Text-Based Question-Answer Classification and Optimization Analysis

⭐ **In this project, the ytu-ce-cosmos/Turkish-Gemma-9b-T1 model of our Cosmos team, which I am a part of, was used.** ⭐

The main purpose of the project is to create a synthetic dataset using local large language models (LLM) and to mathematically analyze the performance of different optimization algorithms (GD, SGD, Adam) on this data .

## 📑 Project Overview

The system consists of three main stages:

**• Data Generation:** Generation of 100 training and 100 test samples (Question + Good Answer + Bad Answer) with the Turkish-Gemma-9b-T1 model.

**• Vectorization:** Converting texts into high-dimensional semantic representation (embedding) vectors with the turkish-e5-large model.

**• Regression and Optimization:** Learning weight parameters and comparing algorithms using a tanh model similar to logistic regression.

## 🏗️ Technical Architecture

### 1. Data Preparation (Data Generation)
Questions were generated on verbal-heavy subjects such as History, Geography, and Technology using the language model.

**• Good Answer (+1):** Answers that are scientifically correct and contain precise statements.

**• Bad Answer (-1):** Answers that are too short, nonsensical, or contain incorrect information.

### 2. Model Structure
The model uses the concatenation of question and answer embedding vectors as input. Its mathematical formulation is as follows:

**c\output\c​=tanh(w⋅x)**

Here, x represents the input vector, and w represents the parameters to be learned.

### 3. Optimization Strategies
Within the scope of the project, three algorithms were compared for 5 different initial w values:

**• GD (Batch Gradient Descent):** Makes stable but slow updates using the entire dataset.

**• SGD (Stochastic Gradient Descent):** Follows a noisy but fast path by selecting a random sample at each step.

**• Adam:** Provides the fastest convergence by combining momentum and adaptive learning rate.

## 📊 Analysis and Visualization

### Performance Graphs:
• Training and test success; analyzed according to Time vs. Loss and Epoch vs. Accuracy criteria.

• The Adam algorithm was the method that learned the fastest and reached the lowest loss value by its nature.

• The effect of the initial weights (w) on the convergence speed was observed, especially in sudden jumps.

### Trajectory Analysis with t-SNE
• The path followed by the weight parameters (w1:t) during the optimization process was reduced to 2 dimensions and visualized with the t-SNE algorithm.

• It has been proven that the algorithms tend towards similar minimum regions despite starting from different initial points.

## 🛠️ Technologies Used

**• Language:** Python 

**• LLM & Embedding:** * ytu-ce-cosmos/Turkish-Gemma-9b-T1 & ytu-ce-cosmos/turkish-e5-large

**• Libraries:** llama-cpp-python, SentenceTransformer, NumPy, Matplotlib, scikit-learn, PyTorch

**• Platform:** Google Colab (GPU Supported)
