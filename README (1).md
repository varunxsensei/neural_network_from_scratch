# 🧠 Neural Network from Scratch in NumPy

This project implements a fully functioning feedforward neural network from the ground up — **without using any deep learning libraries** like TensorFlow or PyTorch. Just pure NumPy.

It includes:
- Dense layers (fully connected)
- ReLU and Softmax activations
- Categorical Cross-Entropy loss
- Forward pass for multi-class classification using synthetic spiral data

---

## 🚀 Features

- ✅ Built from scratch using NumPy
- ✅ Layered, class-based architecture
- ✅ Clean and modular file structure
- ✅ End-to-end forward pass logic
- 🚧 Backpropagation planned in future (in a separate repo)

---

## 📂 File Structure

```
nn_numpy/
│
├── dense_layers_with_activation.py     # Main script to run the forward pass
├── dot_product.py                      # Manual implementation of dot product with bias
├── layers.py                           # Layer and activation classes
├── softmax.py                          # Softmax activation logic
├── spiral_data.py                      # Generates spiral classification dataset
├── __init__.py                         # Marks the directory as a Python module
├── __pycache__/                        # Compiled Python cache files
└── README.md                           # Project overview (this file)
```

---

## 🧪 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/varunxsensei/neural_network_from_scratch.git
   cd neural_network_from_scratch/nn_numpy
   ```

2. Install dependencies:
   ```bash
   pip install numpy
   ```

3. Run the script:
   ```bash
   python dense_layers_with_activation.py
   ```

You’ll see the output of the network after the ReLU and Softmax activations, as well as the calculated loss.

---

## 📚 What You'll Learn

- The math behind neural network layers (dot product + bias)
- How ReLU and Softmax functions work
- How to compute Categorical Cross-Entropy loss manually
- How to build and structure neural networks from scratch

---

## 🛠 Next Steps

The next part of this project (planned for a new repo):
- 🔁 Backpropagation logic
- 🎯 Gradient Descent updates
- 📈 Model training loop
- 📊 Accuracy and performance tracking

---

## 👨‍💻 Author

Made with ❤️ by [Varun](https://github.com/varunxsensei)  
Aspiring ML & NLP Engineer | Deep Learning Enthusiast

---

## 📄 License

MIT License — feel free to use, modify, and share!