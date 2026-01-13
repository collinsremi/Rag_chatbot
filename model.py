import streamlit as st
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import PyPDF2

# Try to import docx, but make it optional
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    st.warning("⚠️ python-docx not installed. DOCX files will not be supported. Install with: pip install python-docx")

# Page config
st.set_page_config(page_title="AI Assistance", page_icon="📚", layout="wide")

# Directory for saving models and documents
MODEL_SAVE_DIR = Path("saved_models")
DOCS_DIR = Path("documents")  # Local folder for your documents
EMBEDDINGS_DIR = Path("embeddings")
MODEL_SAVE_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'current_model_name' not in st.session_state:
    st.session_state.current_model_name = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = None
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []
if 'chunk_embeddings' not in st.session_state:
    st.session_state.chunk_embeddings = None
if 'rag_enabled' not in st.session_state:
    st.session_state.rag_enabled = False
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'embeddings_file' not in st.session_state:
    st.session_state.embeddings_file = EMBEDDINGS_DIR / "embeddings.pkl"

# Default models
DEFAULT_MODEL = "google/flan-t5-small"
DEFAULT_MODEL_DISPLAY = "FLAN-T5-Small (300MB)"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Small, fast embedding model

def get_model_path(model_name):
    """Get the local path for a model"""
    safe_name = model_name.replace("/", "_")
    return MODEL_SAVE_DIR / safe_name

def is_model_saved(model_name):
    """Check if model is already saved locally"""
    model_path = get_model_path(model_name)
    return model_path.exists() and (model_path / "config.json").exists()

@st.cache_resource
def load_t5_model(model_name):
    """Load the T5 model and tokenizer"""
    model_path = get_model_path(model_name)
    
    if is_model_saved(model_name):
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = T5ForConditionalGeneration.from_pretrained(str(model_path))
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))
    
    return model, tokenizer

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model for embeddings"""
    return SentenceTransformer(EMBEDDING_MODEL)

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    if not DOCX_AVAILABLE:
        raise ImportError("python-docx is not installed")
    
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    
    return chunks

def process_document(file_path):
    """Process document from file path and extract text"""
    file_extension = file_path.suffix.lower()
    
    try:
        if file_extension == '.pdf':
            text = extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            text = extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            text = extract_text_from_txt(file_path)
        else:
            return None, "Unsupported file format"
        
        # Chunk the text
        chunks = chunk_text(text)
        
        if not chunks:
            return None, "No text extracted from document"
        
        return chunks, None
    except Exception as e:
        return None, str(e)

def scan_documents_folder():
    """Scan the documents folder and return list of supported files"""
    supported_extensions = ['.pdf', '.txt']
    if DOCX_AVAILABLE:
        supported_extensions.append('.docx')
    
    files = []
    
    if DOCS_DIR.exists():
        for file_path in DOCS_DIR.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
    
    return files

def load_saved_embeddings():
    """Load previously saved embeddings"""
    embeddings_file = st.session_state.embeddings_file
    
    if embeddings_file.exists():
        try:
            with open(embeddings_file, 'rb') as f:
                data = pickle.load(f)
                st.session_state.document_chunks = data['chunks']
                st.session_state.chunk_embeddings = data['embeddings']
                st.session_state.rag_enabled = True
                st.session_state.documents_processed = True
                return True
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            return False
    return False

def save_embeddings():
    """Save embeddings to disk for faster loading"""
    embeddings_file = st.session_state.embeddings_file
    
    try:
        data = {
            'chunks': st.session_state.document_chunks,
            'embeddings': st.session_state.chunk_embeddings
        }
        with open(embeddings_file, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        st.error(f"Error saving embeddings: {str(e)}")
        return False

def process_all_documents():
    """Process all documents in the documents folder"""
    files = scan_documents_folder()
    
    if not files:
        return False, "No documents found in 'documents' folder"
    
    all_chunks = []
    processed_files = []
    
    for file_path in files:
        chunks, error = process_document(file_path)
        if error:
            st.warning(f"Error processing {file_path.name}: {error}")
        else:
            all_chunks.extend(chunks)
            processed_files.append(file_path.name)
    
    if not all_chunks:
        return False, "No text could be extracted from documents"
    
    st.session_state.document_chunks = all_chunks
    
    # Create embeddings for all chunks
    embeddings = st.session_state.embedding_model.encode(all_chunks)
    st.session_state.chunk_embeddings = embeddings
    
    # Save embeddings to disk
    save_embeddings()
    
    st.session_state.rag_enabled = True
    st.session_state.documents_processed = True
    
    return True, processed_files

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve the most relevant chunks for a query"""
    if not st.session_state.document_chunks or st.session_state.chunk_embeddings is None:
        return []
    
    # Embed the query
    query_embedding = st.session_state.embedding_model.encode([query])[0]
    
    # Calculate cosine similarity
    similarities = np.dot(st.session_state.chunk_embeddings, query_embedding)
    
    # Get top k most similar chunks
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_chunks = [st.session_state.document_chunks[i] for i in top_indices]
    scores = [similarities[i] for i in top_indices]
    
    return list(zip(relevant_chunks, scores))

def is_greeting_or_small_talk(text):
    """Detect if the input is a greeting or small talk"""
    greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
                 'good evening', 'what\'s up', 'whats up', 'sup', 'howdy']
    small_talk = ['how are you', 'how r u', 'what do you mean', 'tell me more', 
                  'okay', 'ok', 'thanks', 'thank you']
    
    text_lower = text.lower().strip()
    
    # Check for exact matches or if the text starts with greeting
    for greeting in greetings + small_talk:
        if text_lower == greeting or text_lower.startswith(greeting):
            return True
    
    # Check if it's a very short message (likely small talk)
    if len(text_lower.split()) <= 2:
        return True
    
    return False

def handle_greeting_or_small_talk(prompt):
    """Generate appropriate responses for greetings and small talk"""
    responses = {
        'greeting': "Hello! I'm here to help you with questions about your documents. What would you like to know?",
        'how_are_you': "I'm functioning well, thank you! I'm ready to answer questions about your documents. What can I help you with?",
        'what_do_you_mean': "I'd be happy to clarify! Could you please ask a specific question about the content in your documents? For example, you could ask about specific topics, definitions, or information contained in the uploaded files.",
        'thanks': "You're welcome! Feel free to ask any other questions about your documents.",
        'default': "I'm here to answer questions about your documents. Please ask me something specific about the content in your uploaded files!"
    }
    
    prompt_lower = prompt.lower().strip()
    
    if any(word in prompt_lower for word in ['hi', 'hello', 'hey', 'greetings']):
        return responses['greeting']
    elif 'how are you' in prompt_lower or 'how r u' in prompt_lower:
        return responses['how_are_you']
    elif 'what do you mean' in prompt_lower or 'what' in prompt_lower:
        return responses['what_do_you_mean']
    elif any(word in prompt_lower for word in ['thanks', 'thank']):
        return responses['thanks']
    else:
        return responses['default']

def generate_response(prompt, use_rag=True, max_context=3):
    """Generate response using T5 with optional RAG"""
    
    # Handle greetings and small talk separately
    if is_greeting_or_small_talk(prompt):
        return handle_greeting_or_small_talk(prompt)
    
    if use_rag and st.session_state.rag_enabled:
        # Retrieve relevant context from documents
        relevant_chunks = retrieve_relevant_chunks(prompt, top_k=max_context)
        
        # Check if retrieved chunks are actually relevant (similarity threshold)
        if relevant_chunks and relevant_chunks[0][1] > 0.3:  # Minimum similarity score
            context = "\n\n".join([chunk for chunk, score in relevant_chunks])
            
            # Better prompt engineering for T5
            input_text = f"""Based on the following context, answer the question. If the context doesn't contain the answer, say "I don't have information about that in the documents."

Context: {context}

Question: {prompt}

Answer:"""
        else:
            # No relevant context found - provide helpful fallback
            input_text = f"""You are a helpful assistant. The user asked: "{prompt}"
            
Provide a brief, helpful response. If this is a greeting, respond warmly. If it's a question without enough context, ask for clarification about what specific information they need from the documents.

Response:"""
    else:
        # Regular conversation without RAG
        input_text = f"Answer this question naturally and helpfully: {prompt}"
    
    # Tokenize and generate
    inputs = st.session_state.tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    )
    
    with torch.no_grad():
        outputs = st.session_state.model.generate(
            inputs.input_ids,
            max_length=200,
            min_length=15,
            num_beams=5,
            temperature=0.8,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True
        )
    
    response = st.session_state.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up response
    response = response.strip()
    
    # Fallback for poor responses
    if not response or len(response.strip()) < 5 or response.count(' ') < 3:
        response = "I apologize, but I couldn't generate a proper response. Could you please ask a more specific question about the content in your documents?"
    
    return response

# Auto-load models on first run
if not st.session_state.model_loaded:
    with st.spinner("Loading models... This may take a minute on first run."):
        try:
            model, tokenizer = load_t5_model(DEFAULT_MODEL)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.current_model_name = DEFAULT_MODEL_DISPLAY
            
            # Load embedding model
            st.session_state.embedding_model = load_embedding_model()
            
            st.session_state.model_loaded = True
            
            # Try to load saved embeddings first
            embeddings_loaded = load_saved_embeddings()
            
            # If no saved embeddings, automatically process documents if available
            if not embeddings_loaded:
                doc_files = scan_documents_folder()
                if doc_files:
                    with st.spinner("Processing documents automatically..."):
                        success, result = process_all_documents()
                        if success:
                            st.success(f"✅ Auto-processed {len(result)} file(s)!")
                            st.success(f"📊 {len(st.session_state.document_chunks)} chunks created")
                        else:
                            st.warning(f"⚠️ Could not process documents: {result}")
            else:
                st.success("✅ Loaded saved embeddings from previous session!")
            
            st.rerun()
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()

# Sidebar
st.sidebar.title("⚙️ Settings")

# Model info
st.sidebar.success(f"🤖 T5 Model: {DEFAULT_MODEL_DISPLAY}")
st.sidebar.info(f"🔍 Embeddings: {EMBEDDING_MODEL}")

st.sidebar.markdown("---")

# Document management section
st.sidebar.subheader("📄 Document Management")

# Show documents folder info
doc_files = scan_documents_folder()
st.sidebar.info(f"📁 Folder: `documents/`")
st.sidebar.info(f"📚 {len(doc_files)} file(s) found")

if doc_files:
    with st.sidebar.expander("View Files"):
        for file_path in doc_files:
            st.write(f"• {file_path.name}")

# Process documents button
if doc_files:
    if not st.session_state.documents_processed:
        if st.sidebar.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                success, result = process_all_documents()
                if success:
                    st.sidebar.success(f"✅ Processed {len(result)} file(s)!")
                    st.sidebar.success(f"📊 {len(st.session_state.document_chunks)} chunks created")
                    st.rerun()
                else:
                    st.sidebar.error(f"❌ {result}")
    else:
        st.sidebar.success("✅ Documents processed")
        
        # Reprocess button
        if st.sidebar.button("🔄 Reprocess Documents"):
            st.session_state.documents_processed = False
            st.rerun()
        
        # Clear embeddings
        if st.sidebar.button("🗑️ Clear Processed Data"):
            st.session_state.document_chunks = []
            st.session_state.chunk_embeddings = None
            st.session_state.rag_enabled = False
            st.session_state.documents_processed = False
            
            # Delete embeddings file
            if st.session_state.embeddings_file.exists():
                st.session_state.embeddings_file.unlink()
            
            st.rerun()
else:
    st.sidebar.warning("⚠️ No documents found")
    st.sidebar.info("Add PDF, TXT, or DOCX files to the `documents/` folder")

# Show RAG status
st.sidebar.markdown("---")
if st.session_state.rag_enabled:
    st.sidebar.success(f"✅ RAG Enabled")
    st.sidebar.info(f"📚 {len(st.session_state.document_chunks)} chunks loaded")
else:
    st.sidebar.warning("⚠️ RAG Disabled")
    st.sidebar.info("Process documents to enable RAG")

# RAG settings
st.sidebar.subheader("🔧 RAG Settings")
num_chunks = st.sidebar.slider(
    "Retrieved Chunks",
    min_value=1,
    max_value=5,
    value=3,
    help="Number of relevant chunks to retrieve for each query"
)

# RAG is always enabled if documents are processed
use_rag = st.session_state.rag_enabled

st.sidebar.markdown("---")

# Clear conversation button
if st.sidebar.button("Clear Conversation"):
    st.session_state.conversation_history = []
    st.rerun()

# Main interface
st.title("🤖 AI Assistance")
st.markdown("Ask questions about our services and how it works")

# Display RAG status
if st.session_state.rag_enabled:
    st.info("🟢 RAG Active")
else:
    st.warning("🟡 RAG Inactive")

# Display conversation history
chat_container = st.container()
with chat_container:
    for exchange in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.write(exchange['user'])
        with st.chat_message("assistant"):
            st.write(exchange['assistant'])
            if 'sources' in exchange and exchange['sources']:
                with st.expander("📑 View Sources"):
                    for i, (chunk, score) in enumerate(exchange['sources'], 1):
                        st.markdown(f"**Source {i}** (relevance: {score:.2f})")
                        st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)

# User input
user_input = st.chat_input("Ask a specific question about your documents (e.g., 'What is X?', 'Explain Y', 'Summarize Z')...")

if user_input:
    if st.session_state.model is None:
        st.error("Model not loaded! Please refresh the page.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get relevant sources if RAG is enabled
                sources = []
                if use_rag and st.session_state.rag_enabled:
                    sources = retrieve_relevant_chunks(user_input, top_k=num_chunks)
                
                response = generate_response(user_input, use_rag=use_rag, max_context=num_chunks)
                st.write(response)
                
                # Show sources
                if sources:
                    with st.expander("📑 View Sources"):
                        for i, (chunk, score) in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** (relevance: {score:.2f})")
                            st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        
        # Update conversation history
        st.session_state.conversation_history.append({
            'user': user_input,
            'assistant': response,
            'sources': sources if sources else []
        })
        
        st.rerun()

# Display model info
if st.session_state.model is not None:
    doc_status = f"📚 {len(st.session_state.document_chunks)} chunks" if st.session_state.rag_enabled else "No documents processed"
    st.info(f"🟢 System Ready | {doc_status} | Retrieving top {num_chunks} chunks per query")
