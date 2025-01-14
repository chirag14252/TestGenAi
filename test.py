import os
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import numpy as np
from PyPDF2 import PdfReader
import streamlit as st
import pickle
import time

#load the env variables
# load_dotenv()

api_key = st.secrets["OPENAI_API_KEY"];
ass_id = st.secrets["ASSISTANT_ID"];
# Initialize the OpenAI client with the provided API key
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# # Define the assistant ID for the OpenAI API
# ass_id = os.getenv("ASSISTANT_ID")

script_dir = os.path.dirname(os.path.abspath(__file__))

# Define paths for game-specific resources
GAME_PATHS = {
    "DTC": {
        "pdf_directory": os.path.join(script_dir, "dir/dtc/"),  # Directory containing PDF files for DTC
        "index_path": os.path.join(script_dir, "faiss/dtc/faiss_index.idx"),  # Path to the FAISS index for DTC
        "mapping_path": os.path.join(script_dir, "faiss/dtc/file_mapping.txt"),  # Path to the file mapping for DTC
        "documents_path": os.path.join(script_dir, "faiss/dtc/documents.pkl")  # Path to the documents for DTC
    },
}

# Function to initialize session state variables for Streamlit
def initialize_session_state():
    if 'documents' not in st.session_state:
        st.session_state.documents = None  # Initialize documents to None
    if 'index' not in st.session_state:
        st.session_state.index = None  # Initialize index to None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []  # Initialize conversation history as an empty list
    if 'query_input' not in st.session_state:
        st.session_state.query_input = ""  # Initialize query input to an empty string
    if 'thread_id' not in st.session_state:
        try:
            thread = client.beta.threads.create()  # Create a new thread for conversation
            st.session_state.thread_id = thread.id  # Store the thread ID in session state
        except Exception as e:
            print(f"Error creating initial thread: {e}")  # Log any errors during thread creation
            st.session_state.thread_id = None  # Set thread ID to None on error
    if 'selected_game' not in st.session_state:
        st.session_state.selected_game = "DTC"  # Default selected game is DTC
    if 'clear_input_flag' not in st.session_state:
        st.session_state.clear_input_flag = False
    if 'new_file_uploaded' not in st.session_state:
        st.session_state.new_file_uploaded = False
   

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)  # Create a PDF reader object
    text = ""
    for page in reader.pages:
        text += page.extract_text()  # Extract text from each page
    return text  # Return the extracted text

# Function to generate embeddings using OpenAI
def get_embeddings(texts):
    response = client.embeddings.create(
        input=texts,  # Input texts for embedding
        model="text-embedding-ada-002"  # Specify the model to use
    )
    return [entry.embedding for entry in response.data]  # Return the embeddings

def clear_chat():
    st.session_state.conversation_history = []  # Clear the conversation history
    st.rerun()  # Rerun the Streamlit app to reflect changes
    # Create a new thread when clearing chat
    try:
        thread = client.beta.threads.create()  # Create a new thread
        st.session_state.thread_id = thread.id  # Store the new thread ID
    except Exception as e:
        print(f"Error creating new thread: {e}")  # Log any errors during thread creation

# Function to split text into smaller chunks of approximately max_chunk_size words
def chunk_text(text, max_chunk_size=3000):
    """Split text into smaller chunks of approximately max_chunk_size words"""
    words = text.split()  # Split the text into words
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)  # Add word to the current chunk
        current_size += 1  # Increment the current size
        
        if current_size >= max_chunk_size:  # Check if the current chunk size exceeds the limit
            chunks.append(' '.join(current_chunk))  # Add the current chunk to the list of chunks
            current_chunk = []  # Reset the current chunk
            current_size = 0  # Reset the current size
            
    if current_chunk:  # If there are remaining words in the current chunk
        chunks.append(' '.join(current_chunk))  # Add the last chunk
        
    return chunks  # Return the list of chunks

# Load all PDFs from a directory and extract their text
def load_pdfs_from_directory(directory):
    pdf_texts = []  # List to hold extracted texts
    pdf_files = []  # List to hold file names
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdf"):  # Check if the file is a PDF
            file_path = os.path.join(directory, file_name)  # Get the full file path
            print(f"Processing {file_name}...")  # Log the processing of the file
            text = extract_text_from_pdf(file_path)  # Extract text from the PDF
            # Split the text into chunks
            chunks = chunk_text(text)  # Chunk the extracted text
            for chunk in chunks:
                pdf_texts.append(chunk)  # Add chunk to the list of texts
                pdf_files.append(f"{file_name}_chunk_{len(pdf_texts)}")  # Create a unique file name for the chunk
    return pdf_texts, pdf_files  # Return the lists of texts and file names

def create_faiss_index(documents):
    """
    Create and return a FAISS index from document embeddings
    """
    print("Generating embeddings...")  # Log the start of embedding generation
    document_embeddings = get_embeddings(documents)  # Get embeddings for the documents
    
    dimension = len(document_embeddings[0])  # Get the dimension of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index
    index.add(np.array(document_embeddings).astype('float32'))  # Add embeddings to the index
    print(f"FAISS index created with {index.ntotal} entries.")  # Log the number of entries in the index
    
    return index  # Return the created index

def save_index_and_mapping(index, file_names, index_path, mapping_path):
    """
    Save FAISS index and file mapping to disk
    """
    faiss.write_index(index, index_path)  # Save the FAISS index to disk
    print(f"FAISS index saved as '{index_path}'.")  # Log the save operation
    
    with open(mapping_path, "w") as f:
        for idx, name in enumerate(file_names):
            f.write(f"{idx}: {name}\n")  # Write the mapping of index to file names
    print(f"File mapping saved as '{mapping_path}'.")  # Log the save operation

def check_search_milestones():
    # print("history", st.session_state.conversation_history)
    count = len(st.session_state.conversation_history)
    print("count", count)
    if count >= 3 and count <= 5:
        return "Add runtime examples to verify the test cases. "
    
    elif count >=6   and count <= 7:  # After 2 more searches
        return "Add experiment shift checks"
    return ""

# Querying the FAISS index (Example)
def query_index(query, conversation_history, k=9, threshold=0.7):
    """Query using Assistant API with threads"""
    try:
        
        # Get query embedding and search FAISS index (existing code)
        query_embedding = get_embeddings([query])[0]
        distances, indices = st.session_state.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        # Collect relevant chunks (existing code)
        relevant_chunks = []
        for i in range(k):
            if distances[0][i] < threshold:
                chunk_index = indices[0][i]
                relevant_chunks.append(st.session_state.documents[chunk_index])
        
        if not relevant_chunks:
            return "No relevant documents found. Please be more specific."
        
        
        check_search_milestones();
        
        rules = [
          "remove comment section in the format",
          "be specific about the test case"
        ]
        if len(st.session_state.conversation_history) <= 3:
            rules.append("do not include the existing test cases like {st.session_state.conversation_history}")

        context = f"""
        Test Case Template Format:
        - Title: [Brief description of the test case]
        - Preconditions: [Any setup required before executing the test]
        - Steps: [Detailed steps to execute the test]
        - Expected Result: [What should happen after executing the steps]
        - Status: [Pass/Fail]
        {rules}
        {check_search_milestones()}
        Specification Content:
        {' '.join(relevant_chunks)}
        """
        
        # Add message to thread with enhanced context
        message = client.beta.threads.messages.create(
            thread_id=st.session_state.thread_id,
            role="user",
            content=f"Using this test case format and context, generate test cases for: {query}\n\n{context}"
        )
        
        # Rest of your existing code remains the same
        run = client.beta.threads.runs.create(
            thread_id=st.session_state.thread_id,
            assistant_id=ass_id
        )
        
        # Wait for completion with timeout
        start_time = time.time()
        while True:
            if time.time() - start_time > 30:  # 30 second timeout
                return "Request timed out. Please try again."
                
            run_status = client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread_id,
                run_id=run.id  # Retrieve the run status
            )
            
            if run_status.status == 'completed':  # Check if the run is completed
                break
            elif run_status.status in ['failed', 'cancelled', 'expired']:  # Handle failed runs
                return f"Error: Run {run_status.status}. Please try again."
            
            time.sleep(1)  # Wait before checking the status again
        
        # Get the latest message
        messages = client.beta.threads.messages.list(
            thread_id=st.session_state.thread_id  # List messages in the thread
        )
        
        # Return the assistant's response
        return messages.data[0].content[0].text.value  # Return the response text

    except Exception as e:
        print(f"Error in query_index: {e}")  # Log any errors during querying
        return f"An error occurred: {str(e)}"  # Return the error message

def format_conversation_history(history):
    formatted = []  # List to hold formatted conversation history
    for entry in history:
        formatted.append(f"User: {entry['question']}")  # Format user question
        formatted.append(f"Assistant: {entry['answer']}")  # Format assistant answer
    return "\n".join(formatted)  # Return the formatted history as a string

def prep_faiss_index():
    print("prep_faiss_index called")
    game_paths = GAME_PATHS[st.session_state.selected_game]  # Get paths for the selected game
    if st.session_state.documents is None or st.session_state.index is None:  # Check if documents or index are not loaded
        if os.path.exists(game_paths["index_path"]) and os.path.exists(game_paths["documents_path"]):  # Check if saved files exist
            print(f"Loading saved index and documents for {st.session_state.selected_game}...")  # Log loading operation
            st.session_state.index = faiss.read_index(game_paths["index_path"])  # Load the FAISS index
            
            with open(game_paths["documents_path"], "rb") as f:
                st.session_state.documents = pickle.load(f)  # Load the documents
            print("Loaded saved files successfully!")  # Log successful loading
        else:
            print(f"Creating new index and documents for {st.session_state.selected_game}...")  # Log creation operation
            documents, file_names = load_pdfs_from_directory(game_paths["pdf_directory"])  # Load PDFs from directory
            print(f"Loaded {len(documents)} PDF files.")  # Log the number of loaded PDFs

            index = create_faiss_index(documents)  # Create a new FAISS index
            save_index_and_mapping(index, file_names, 
                                 game_paths["index_path"], 
                                 game_paths["mapping_path"])  # Save the index and mapping
            
            with open(game_paths["documents_path"], "wb") as f:
                pickle.dump(documents, f)  # Save the documents
            
            st.session_state.documents = documents  # Store documents in session state
            st.session_state.index = index  # Store index in session state
    
    return st.session_state.documents, st.session_state.index  # Return documents and index

# Function to clear input
def clear_input():
    print("clear input called")
    print("query input", st.session_state.query_input)
    print("query input last char", st.session_state.query_input[-1])
   

def generate_test_cases_query(query_input):
    prep_faiss_index();
    print("generate_test_cases_query called")
    if query_input.strip():  # Check if there's valid input
        result = query_index(query_input, st.session_state.conversation_history)
        st.session_state.conversation_history.append({
            "question":query_input,
            "answer": result
        })
        # clear_input()

def reset_game_state():
    """Reset session state when game changes"""
    st.session_state.documents = None  # Reset documents
    st.session_state.index = None  # Reset index
    st.session_state.conversation_history = []  # Clear conversation history
    try:
        thread = client.beta.threads.create()  # Create a new thread
        st.session_state.thread_id = thread.id  # Store the new thread ID
    except Exception as e:
        print(f"Error creating new thread: {e}")  # Log any errors during thread creation
        st.session_state.thread_id = None  # Set thread ID to None on error



# Streamlit UI
def streamlit_ui_new():
    st.set_page_config(page_title="Test Case Generator", page_icon="📚", layout="wide", initial_sidebar_state="collapsed")  # Set up the Streamlit page

   # Custom CSS for styling
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #f8f8f8, #f0e6f6, #e6f0f6);
            background-size: 400% 400%;
            animation: gradientBG 15s ease infinite;
            position: relative;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .container {
            text-align: center;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            width: 90vw;
            margin: 0 auto;
        }

        h1, p, .description, .assistant-message {
            color: #000;
        }

        h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }

        p {
            font-size: 2rem;
            margin-bottom: 1rem;
        }

        .description {
            font-size: 1.5rem;
            margin-bottom: 2rem;
        }

        .button {
            background: linear-gradient(135deg, #e0e0e0, #d0d0f0, #c0e0f0);  /* Adjusted to match body background */
            color: black;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 25px;
            font-size: 1.2rem;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }

       

        .assistant-message {
            background-color: white;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
                
        [data-testid='stFileUploader'] {
            color: black;            
        }

        [data-testid='stFileUploader'] label {
            color: black;
            font-style: italic;
        }

       
      
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="container">
        <h1 style="color: black; font-size: 4rem;">Test Case Generator</h1>
        <div class="description">
            Welcome to the Test Case Generator! Our platform helps you create comprehensive test cases with ease.<br>
            Upload the spec pdf and voila! you have test cases.<br>
            <i><b>Note : 6 to 10 test cases are generated at a time. Click Generate test case again to see more results.</b></i><br>
        </div>
    </div>
    """, unsafe_allow_html=True)

    
    # Game selector
    col1, col2, col3 = st.columns([1, 2, 1])  # Create columns for layout
    # with col2:
    #     st.markdown("<h3 style='text-align: center;'>Select Game</h3>", unsafe_allow_html=True)  # Game selection header
    #     selected_game = st.selectbox(
    #         "",
    #         ["DTC"],  # Options for game selection
    #         key="selected_game",
    #         on_change=lambda: reset_game_state()  # Reset state on game change
    # )

    initialize_session_state()  # Initialize session state variables
    prep_faiss_index()  # Prepare the FAISS index


    #chat UI
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat container
    for entry in st.session_state.conversation_history:
        st.markdown(f'<div class="assistant-message"><strong>QA support:</strong> {entry["answer"]}</div>', unsafe_allow_html=True)  # Display assistant message
    st.markdown('</div>', unsafe_allow_html=True)  # Close chat container

    # Input Area
    query_input = st.session_state.query_input
    if st.session_state.clear_input_flag:
        st.session_state.query_input = ""
        st.session_state.clear_input_flag = False
    else:
        st.session_state.query_input = query_input# Input for user query
    # query = st.text_input("Enter the spec name :", key="query_input", placeholder="Enter the name of the spec", on_change=clear_input)  
   
    # Create two columns for the buttons
    col1, spacing, col2 = st.columns([0.3, 0.01, 2.0])  # Layout for buttons

    
       # Place buttons in separate columns
    with col1:
        # Determine button text based on conversation history
        button_text = "Generate Test Cases"
        
        # Add custom CSS to change button color and height
        st.markdown("""
        <style>
        .stButton>button {
            background: linear-gradient(135deg, #e0e0e0, #d0d0f0, #c0e0f0);  /* Adjusted to match body background */
            color: black;
            height: 80px;  /* Set a fixed height for the button */
            margin-top: 10px;  /* Add margin to move the button down */
            border: none;  /* Remove border */
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #d0d0d0, #c0c0e0, #b0d0e0);  /* Change hover background */
            color: black;
        }
        </style>
        """, unsafe_allow_html=True)
        
        if st.button(button_text, use_container_width=True, key="generate_button") and query_input:  # Submit button
            print(f"Submitting query: {query_input}")  # Log the submitted query
            # Create a container for the spinner and text
            with st.spinner("Thinking how a QA will think like..."): 
                generate_test_cases_query("generate test cases for " + query_input) 
                st.session_state.clear_input_flag = True 
                st.rerun()  # Rerun
    
    with col2:
        uploaded_file = st.file_uploader(
            "Upload Spec PDF",
            type=['pdf'],
            label_visibility="visible",
            help="Upload specification PDF"
        )

        if uploaded_file is not None:
            try:
                game_paths = GAME_PATHS[st.session_state.selected_game]
                pdf_dir = game_paths["pdf_directory"]
                faiss_dir = "faiss/dtc"
                # Remove existing PDF files except the uploaded one
                for file_name in os.listdir(pdf_dir):
                    if file_name.endswith(".pdf") and file_name != uploaded_file.name:
                        file_path = os.path.join(pdf_dir, file_name)
                        os.remove(file_path)
                for file_name in os.listdir(faiss_dir):
                    if file_name.endswith(".idx") or file_name.endswith(".txt") or file_name.endswith(".pkl"):
                        file_path = os.path.join(faiss_dir, file_name)
                        os.remove(file_path)
                
                file_path = os.path.join(pdf_dir, uploaded_file.name)
                
                # Check if file already exists
                if not os.path.exists(file_path):
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                        print("file saved")
                    st.session_state.documents = None
                    st.session_state.index = None
                    st.session_state.conversation_history = []
                    st.session_state.query_input = uploaded_file.name.replace('.pdf', '')
                    # Do not call prep_faiss_index here
                else:
                    st.session_state.query_input = uploaded_file.name.replace('.pdf', '')
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
    
    # with col3:
    #     if st.button("Clear Chat", use_container_width=False):  # Clear chat button
    #         clear_chat()  # Clear the chat history
    # st.markdown('</div>', unsafe_allow_html=True) # Close chat container


    # Advanced Settings
    with st.expander("Advanced Search Settings"):  # Expandable section for advanced settings
        k = st.slider("Number of relevant spec chunks to check, keep it high if you are not satisfied with answer", min_value=1, max_value=20, value=9)  # Slider for number of chunks
        threshold = st.slider("Similarity threshold, keep it high for higher chance of getting answer", min_value=0.0, max_value=1.0, value=0.7)  # Slider for similarity threshold

# Modify your __main__ block
if __name__ == "__main__":
    streamlit_ui_new()  # Run the Streamlit UI







#what are the set of checks which QAs checks

#After 5 searches , it will give some runtime examples for checking the runtime.
#then for 2 searches it will give runtime checking if not present then ask them to add it and then upload the pdf.
# on 9th search , it will start experiment shift check if not present then ask them to add it and then upload the pdf.

#this to be done by tonight.
#deisgn to be finalized by tonight.



# a simple prompt to where you can check edge cases which are missing from the spec . whith repect to the spec.

#an option to dump the history on to google sheet,





# copy text changes 
#Ui changes
#Runtime changes (give some example runtimes for checks)
#experiment check 
#experiment movement
#runtime movement



#basic check to be checked across 
#remove unnecessary statment , only follow the format.



#creating quality test cases --> need to check and train it
