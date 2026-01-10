# 📚 AI Assistant with RAG (Retrieval-Augmented Generation)

An intelligent question-answering system powered by Google's FLAN-T5 language model and RAG technology. This application allows you to ask questions about your documents and get accurate, context-aware answers based on your own content.

## 🌟 Features

- **🤖 FLAN-T5 Language Model**: Uses Google's FLAN-T5-Small for natural language understanding and generation
- **📄 Document Processing**: Automatically processes PDF, TXT, and DOCX files
- **🔍 Smart Retrieval**: Uses semantic search to find relevant content from your documents
- **💾 Offline Capability**: Works completely offline after initial setup
- **⚡ Fast Performance**: Embeddings are cached for instant loading on subsequent runs
- **🎯 Context-Aware Answers**: Provides answers with source citations and relevance scores
- **💬 Chat Interface**: Clean, intuitive Streamlit interface for natural conversations

## 🎯 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI technique that combines:
1. **Retrieval**: Finding relevant information from your documents
2. **Generation**: Using that information to generate accurate answers

This means the AI answers questions based on YOUR documents, not generic knowledge.

## 📋 Prerequisites

- Python 3.8 or higher
- 500MB free disk space (for models)
- Internet connection (only for first-time setup)

## 🚀 Quick Start

### 1. Installation

Clone or download this repository, then install dependencies:

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with `python-docx`, run:
```bash
pip uninstall docx python-docx -y
pip install python-docx
```

### 2. Add Your Documents

Create a `documents/` folder (or use the auto-created one) and add your files:

```
your_project/
├── app.py
├── requirements.txt
└── documents/          ← Add your files here
    ├── document1.pdf
    ├── document2.txt
    └── document3.docx
```

**Supported formats**: PDF, TXT, DOCX

### 3. Run the Application

```bash
streamlit run app.py
```

The app will:
- ✅ Download models on first run (~400MB total)
- ✅ Automatically detect and process your documents
- ✅ Create embeddings and save them for future use
- ✅ Launch in your browser

### 4. Start Asking Questions!

Simply type your questions in the chat interface and get answers based on your documents.

## 💡 Usage Examples

### Example Questions:

**For a company policy document:**
- "What is the vacation policy?"
- "How do I submit expense reports?"
- "What are the working hours?"

**For research papers:**
- "What methodology was used in this study?"
- "What were the main findings?"
- "Who are the authors?"

**For technical documentation:**
- "How do I install this software?"
- "What are the system requirements?"
- "How do I configure the API?"

**For legal documents:**
- "What are the payment terms?"
- "What is the termination clause?"
- "Who are the parties in this agreement?"

## 🎛️ Interface Guide

### Sidebar Controls

**📄 Document Management**
- View files detected in the `documents/` folder
- See processing status
- Reprocess documents when you add new files
- Clear processed data if needed

**🔧 RAG Settings**
- **Retrieved Chunks**: Adjust how many relevant sections to use (1-5)
  - Lower = Faster, more focused answers
  - Higher = More context, comprehensive answers

**🗑️ Conversation**
- Clear conversation history anytime

### Main Interface

- **Chat Area**: Displays conversation history
- **Input Box**: Type your questions here
- **📑 View Sources**: Click to see which document sections were used for each answer
- **Status Bar**: Shows system status and number of processed chunks

## 📁 Project Structure

```
your_project/
├── app.py                      # Main application file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── documents/                  # Your documents (you create this)
│   ├── file1.pdf
│   ├── file2.txt
│   └── file3.docx
├── saved_models/               # Downloaded models (auto-created)
│   └── google_flan-t5-small/
├── embeddings/                 # Cached embeddings (auto-created)
│   └── embeddings.pkl
```

## 🔧 Configuration

### Models Used

- **T5 Model**: `google/flan-t5-small` (~300MB)
  - Best balance of speed and quality
  - Good for everyday questions
  
- **Embedding Model**: `all-MiniLM-L6-v2` (~90MB)
  - Fast semantic search
  - Accurate relevance matching

### Customization

You can modify these settings in `app.py`:

```python
DEFAULT_MODEL = "google/flan-t5-small"  # Change to flan-t5-base for better quality
EMBEDDING_MODEL = "all-MiniLM-L6-v2"    # Keep this for best performance
```

**Available T5 variants:**
- `google/flan-t5-small` - 300MB, fastest
- `google/flan-t5-base` - 1GB, better quality
- `google/flan-t5-large` - 3GB, best quality (requires more RAM)

## ⚡ Performance Tips

1. **First Run**: Takes 2-5 minutes to download models and process documents
2. **Subsequent Runs**: Loads in 10-15 seconds (embeddings are cached)
3. **Adding Documents**: Click "Reprocess Documents" after adding new files
4. **Chunk Size**: Default 500 words works well for most documents
5. **Retrieved Chunks**: Start with 3, increase if answers lack context

## 🌐 Offline Usage

After the first successful run with internet:

1. ✅ All models are saved locally
2. ✅ Embeddings are cached
3. ✅ **No internet required**
4. ✅ Works completely offline

Perfect for:
- Secure environments
- Air-gapped systems
- Working without internet
- Privacy-sensitive documents

## 🐛 Troubleshooting

### "SentencePiece not found"
```bash
pip install sentencepiece
```

### "python-docx error"
```bash
pip uninstall docx python-docx -y
pip install python-docx
```

### "No documents processed"
- Make sure files are in the `documents/` folder
- Check file formats (PDF, TXT, DOCX only)
- Click "🔄 Process Documents" if auto-processing failed

### "Model loading failed"
- Check internet connection (first run only)
- Ensure you have 500MB free disk space
- Try deleting `saved_models/` folder and restart

### Poor Quality Answers
- Increase "Retrieved Chunks" slider
- Make sure your documents contain relevant information
- Try rephrasing your question more specifically
- Consider upgrading to `flan-t5-base` for better quality

### Slow Performance
- Reduce "Retrieved Chunks" to 1-2
- Use fewer/smaller documents
- Ensure models are saved locally (check `saved_models/` folder)

## 📊 Technical Details

### How It Works

1. **Document Processing**
   - Extracts text from PDF/TXT/DOCX files
   - Splits into 500-word chunks with 50-word overlap
   - Creates vector embeddings for semantic search

2. **Question Answering**
   - Converts your question to a vector embedding
   - Finds most similar document chunks (cosine similarity)
   - Passes relevant chunks + question to T5
   - T5 generates answer based on provided context

3. **Caching**
   - Models cached in `saved_models/`
   - Embeddings cached in `embeddings/embeddings.pkl`
   - Fast loading on subsequent runs

### Technologies Used

- **Streamlit**: Web interface
- **Transformers**: T5 model loading and inference
- **Sentence-Transformers**: Semantic embeddings
- **PyTorch**: Deep learning framework
- **PyPDF2**: PDF text extraction
- **python-docx**: DOCX text extraction
- **NumPy**: Numerical computations

## 🔒 Privacy & Security

- ✅ **100% Local**: All processing happens on your machine
- ✅ **No Cloud**: Documents never leave your computer
- ✅ **No API Calls**: No external services used
- ✅ **Offline Capable**: Works without internet
- ✅ **Your Data**: Complete control over your documents

## 📝 Requirements

See `requirements.txt` for full list. Main dependencies:

- streamlit >= 1.28.0
- transformers >= 4.35.0
- torch >= 2.0.0
- sentence-transformers >= 2.2.0
- PyPDF2 >= 3.0.0
- python-docx >= 0.8.11

## 🤝 Contributing

Suggestions and improvements are welcome! This is a local AI assistant designed for:
- Research
- Document analysis
- Knowledge management
- Study assistance
- Business intelligence

## 📄 License

This project is provided as-is for educational and personal use.

## 🙏 Acknowledgments

- **Google**: FLAN-T5 model
- **Sentence-Transformers**: Embedding models
- **Streamlit**: Amazing UI framework
- **Hugging Face**: Model hosting and Transformers library

## 📞 Support

If you encounter issues:

1. Check the Troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your Python version (3.8+)
4. Check that documents are in the correct folder

## 🚀 Future Enhancements

Potential improvements:
- [ ] Support for more document formats (EPUB, HTML)
- [ ] Multi-language support
- [ ] Custom model selection in UI
- [ ] Conversation memory improvements
- [ ] Export conversations
- [ ] Document upload via interface
- [ ] Better citation formatting

---

**Built with ❤️ using open-source AI**

*Last updated: December 2024*