# Advanced NLP Chatbot Implementation

An intelligent conversational AI system built with advanced Natural Language Processing techniques and Machine Learning algorithms for intent recognition and sentiment-aware responses.

## 🤖 About

This project implements a sophisticated chatbot that leverages NLP techniques to understand user intents and provide contextually appropriate responses. The system incorporates sentiment analysis to deliver more empathetic and human-like interactions.

## 👨‍💻 Creator

**Vaibhav Sharma**
- **Email**: itsvaibhavsharma007@gmail.com
- **GitHub**: [itsvaibhavsharma](https://github.com/itsvaibhavsharma)
- **LinkedIn**: [itsvaibhavsharma](https://linkedin.com/in/itsvaibhavsharma)

**Education**:
- B.Tech, Madhav Institute of Technology & Science (MITS), Gwalior
- B.S. in Data Science, Indian Institute of Technology (IIT) Madras

## 🚀 Features

- **Intent Classification**: Accurately identifies user intentions using Random Forest classifier
- **Sentiment Analysis**: VADER-based sentiment detection for empathetic responses
- **Text Preprocessing**: Advanced NLP pipeline with tokenization, lemmatization, and stopword removal
- **TF-IDF Vectorization**: Efficient feature extraction for text classification
- **Interactive Chat Interface**: Real-time conversational experience
- **Model Persistence**: Save and load trained models for production deployment
- **Confidence Thresholding**: Intelligent fallback for low-confidence predictions
- **Conversation History**: Track and analyze chat sessions

## 🛠️ Technologies Used

### Programming Language
- **Python 3.x**

### Machine Learning & NLP Libraries
- **scikit-learn (sklearn)**
  - `TfidfVectorizer` - Text feature extraction
  - `RandomForestClassifier` - Intent classification
  - `train_test_split` - Data splitting
  - `classification_report`, `accuracy_score` - Model evaluation
- **NLTK (Natural Language Toolkit)**
  - `WordNetLemmatizer` - Text lemmatization
  - `stopwords` - Stopword removal
  - `word_tokenize` - Text tokenization
  - `SentimentIntensityAnalyzer` (VADER) - Sentiment analysis
  - `punkt`, `wordnet`, `omw-1.4`, `vader_lexicon` - Language resources

### Data Processing & Analysis
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **json** - JSON data handling for intents

### Visualization
- **matplotlib.pyplot** - Statistical plotting
- **seaborn** - Advanced data visualization

### Utility Libraries
- **pickle** - Model serialization and persistence
- **random** - Random response selection
- **re** (Regular Expressions) - Text pattern matching
- **ssl** - Secure SSL context for NLTK downloads
- **string** - String operations
- **warnings** - Warning management

### Development Environment
- **Jupyter Notebook**
- **IPython.display** - Interactive display capabilities

## 📊 Model Training

- **Dataset**: 835 patterns across 278 intent categories
- **Evaluation**: Comprehensive classification report with precision, recall, and F1-scores

## 🏗️ Architecture

```
User Input → Text Preprocessing → TF-IDF Vectorization → Random Forest Classifier → Intent Prediction → Response Generation → Sentiment Enhancement → Output
```

### Text Preprocessing Pipeline
1. **Lowercase conversion**
2. **Punctuation removal**
3. **Tokenization** (NLTK)
4. **Stopword removal**
5. **Lemmatization** (WordNet)

### Feature Extraction
- **TF-IDF Vectorization** with 5000 max features
- **Sparse matrix representation** for memory efficiency

### Classification
- **Random Forest Classifier** (100 estimators)
- **Confidence threshold**: 30% minimum for predictions
- **Fallback responses** for low-confidence cases

## 🚀 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/itsvaibhavsharma/green-technology-chatbot

# Install required packages
pip install nltk scikit-learn pandas numpy matplotlib seaborn

# Download NLTK data
python -c "import nltk; nltk.download(['punkt', 'wordnet', 'stopwords', 'omw-1.4', 'vader_lexicon', 'punkt_tab'])"
```

## 💻 Usage

```python
# Basic chatbot interaction
def chat():
    print("Chatbot: Hi there! I'm your AI assistant. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = get_response(user_input)
        print(f"Chatbot: {response}")

# Start chatting
chat()
```

## 📁 Project Structure

```
├── intents.json              # Intent patterns and responses
├── Advanced_NLP_Chatbot.ipynb  # Main implementation notebook
└── README.md                 # Project documentation
```

## 🔮 Future Enhancements

- **Neural Networks**: Implement LSTM/BERT for better context understanding
- **Context Tracking**: Multi-turn conversation memory
- **Entity Recognition**: Named Entity Recognition (NER) integration
- **Spelling Correction**: Automatic typo correction
- **Voice Integration**: Speech-to-text and text-to-speech capabilities
- **Multilingual Support**: Multiple language processing
- **API Deployment**: REST API for web integration

## 📈 Model Metrics

- **Total Intents**: 278 categories
- **Training Patterns**: 835 unique patterns
- **Response Variations**: 567 different responses
- **Test Split**: 80-20 train-test split
- **Evaluation**: Comprehensive classification report with macro and weighted averages

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- NLTK team for comprehensive NLP tools
- scikit-learn community for machine learning algorithms
- Jupyter Project for interactive development environment

---

⭐ **Star this repository if you found it helpful!**