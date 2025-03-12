# Toxic Comment Classification using BERT

A sophisticated machine learning project that uses BERT (Bidirectional Encoder Representations from Transformers) to classify toxic comments. This project provides both a web interface and CLI tools for detecting various types of toxic comments.

## 🌟 Features

- Real-time toxic comment classification
- Interactive web interface using Streamlit
- Command-line interface for batch processing
- Support for multiple toxicity categories
- Visualization of toxicity scores using Plotly
- GPU acceleration support (when available)

## 🛠️ Prerequisites

- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)
- Git

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/commentclassification_using_bert_model.git
   cd commentclassification_using_bert_model
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Web Interface

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser and navigate to the displayed URL (typically http://localhost:8501)
3. Enter text in the input field to get toxicity predictions
4. View the visualization of toxicity scores through an interactive chart

### Command Line Interface

For interactive testing:
```bash
python CLI_interactive_test.py
```

For model training:
```bash
python train.py
```

For running tests:
```bash
python test_model.py
```

## 🏗️ Project Structure

```
├── app.py                  # Streamlit web application
├── CLI_interactive_test.py # Command line interface
├── train.py               # Model training script
├── test_model.py          # Model testing utilities
├── cuda.py               # CUDA availability check
├── requirements.txt       # Project dependencies
├── setup.py              # Package setup configuration
├── src/                  # Source code directory
├── models/               # Saved model checkpoints
└── data/                 # Training and test datasets
```

## 🔧 Model Architecture

The project uses a fine-tuned BERT model (bert-base-uncased) with additional classification layers to detect different types of toxicity in text. The model is implemented using PyTorch and the Transformers library.

Key components:
- BERT base model for text encoding
- Custom classification head for toxicity detection
- Multi-label classification support
- Real-time inference capabilities

## 📊 Performance

The model is trained to classify text into multiple toxicity categories with high accuracy. It can process text in real-time and provides confidence scores for each category of toxicity:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## 💻 Dependencies

Key dependencies include:
- transformers >= 4.35.0
- torch >= 1.9.0
- streamlit >= 1.24.0
- fastapi >= 0.68.0
- plotly >= 5.13.0
- pandas >= 1.3.0
- numpy >= 1.19.0

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Here's how you can contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- The BERT team at Google Research
- The Streamlit team for the excellent web framework
- The PyTorch team for the deep learning framework 