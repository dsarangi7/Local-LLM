import pytest
from main import DocumentProcessor
import re

@pytest.fixture(scope="session")
def processor():
    """Create and initialize a DocumentProcessor instance for testing."""
    proc = DocumentProcessor(
        data_dir="C:/Users/dsara/Desktop/Local LLM/data",
        db_dir="db"
    )
    
    # Load and process documents
    print("\nSetting up test environment...")
    print("Loading and processing documents...")
    documents = proc.load_and_process_documents()
    
    # Initialize vector store
    print("Initializing vector store...")
    proc.initialize_vector_store(documents)
    
    # Initialize RAG
    print("Initializing RAG system...")
    proc.initialize_rag()
    
    return proc

def extract_kwh_value(text):
    """Extract kWh values from text."""
    if not text:
        return None
        
    # Look for patterns like "X kWh", "X.X kWh", "X,XXX kWh"
    patterns = [
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:kWh|kwh|KWH)',  # Matches: 100 kWh, 1,000 kWh, 100.5 kWh
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*kilowatt[- ]hours?', # Matches: 100 kilowatt-hours
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*kW[- ]?h',  # Matches: 100 kW-h, 100 kWh
        r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:thousand)?\s*kW[-\s]?hours?',  # Matches: 100 thousand kW hours
        r'capacity.*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:kWh|kwh|KWH)',  # Matches: capacity of 100 kWh
        r'energy.*?(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:kWh|kwh|KWH)'  # Matches: energy of 100 kWh
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                # Remove commas and convert to float
                return float(matches[0].replace(',', ''))
            except (ValueError, IndexError):
                continue
    return None

def normalize_score(score):
    """Normalize similarity score to 0-1 range."""
    # Convert distance to similarity (smaller distance = higher similarity)
    similarity = 1 / (1 + score)
    return similarity

def test_similarity_search(processor):
    """Test similarity search for installed energy capacity question."""
    query = "How much is the installed energy in kWh"
    
    try:
        results = processor.db.similarity_search_with_score(query, k=5)
        
        # Test that we got results
        assert len(results) > 0, "No results returned from similarity search"
        
        # Print results for debugging
        print("\nSimilarity search results:")
        for i, (doc, score) in enumerate(results, 1):
            normalized_score = normalize_score(score)
            print(f"\nResult {i}:")
            print(f"Raw Score: {score:.4f}")
            print(f"Normalized Score: {normalized_score:.4f}")
            print(f"Content: {doc.page_content[:200]}...")
            kwh = extract_kwh_value(doc.page_content)
            if kwh:
                print(f"Found kWh value: {kwh}")
        
        # Test that the top results are relevant (have kWh mentions)
        found_kwh = False
        for doc, score in results:
            if extract_kwh_value(doc.page_content) is not None:
                found_kwh = True
                break
        
        assert found_kwh, "No kWh values found in top results"
        
        # Test normalized similarity scores are in expected range (0 to 1)
        for _, score in results:
            normalized_score = normalize_score(score)
            assert 0 <= normalized_score <= 1, f"Normalized similarity score {normalized_score} out of expected range [0,1]"
            
    except Exception as e:
        pytest.fail(f"Similarity search failed: {str(e)}")

def test_rag_query(processor):
    """Test RAG query for installed energy capacity."""
    query = "How much is the installed energy in kWh"
    
    try:
        response = processor.query_rag(query)
        
        # Test that we got a response
        assert response, "No response received from RAG system"
        
        # Print response for debugging
        print("\nRAG Response:", response)
        
        # Test that response contains kWh values
        kwh_value = extract_kwh_value(response)
        print(f"Extracted kWh value: {kwh_value}")
        
        # If no direct kWh value found, try to parse numbers with "capacity" or "energy" context
        if kwh_value is None:
            # Look for numbers in the context of capacity or energy
            capacity_pattern = r'(?:capacity|energy).*?(\d+(?:,\d+)*(?:\.\d+)?)'
            matches = re.findall(capacity_pattern, response, re.IGNORECASE)
            if matches:
                try:
                    kwh_value = float(matches[0].replace(',', ''))
                    print(f"Found capacity/energy value: {kwh_value}")
                except (ValueError, IndexError):
                    pass
        
        assert kwh_value is not None, "No kWh or energy capacity value found in response"
        
        # Test that the kWh value is within a reasonable range
        # Adjust these values based on your expected system size
        assert 0 < kwh_value < 1000000, f"kWh value {kwh_value} outside reasonable range"
        
    except Exception as e:
        pytest.fail(f"RAG query failed: {str(e)}")

def test_response_consistency(processor):
    """Test consistency between similarity search and RAG response."""
    query = "How much is the installed energy in kWh"
    
    try:
        # Get similarity search results
        search_results = processor.db.similarity_search_with_score(query, k=5)
        search_kwh_values = [
            extract_kwh_value(doc.page_content)
            for doc, _ in search_results
            if extract_kwh_value(doc.page_content) is not None
        ]
        
        print("\nSearch kWh values found:", search_kwh_values)
        
        # Get RAG response
        rag_response = processor.query_rag(query)
        rag_kwh_value = extract_kwh_value(rag_response)
        
        print("RAG response kWh value:", rag_kwh_value)
        
        # Test that RAG's kWh value matches or is derived from search results
        if search_kwh_values and rag_kwh_value:
            # Check if RAG's value is close to any search result value
            matches_search = any(
                abs(rag_kwh_value - search_value) / search_value < 0.1  # 10% tolerance
                for search_value in search_kwh_values
            )
            assert matches_search, "RAG response kWh value doesn't match any search result"
            
    except Exception as e:
        pytest.fail(f"Response consistency test failed: {str(e)}")

def test_document_metadata(processor):
    """Test that documents have proper metadata."""
    try:
        documents = processor.load_and_process_documents()
        
        print(f"\nTesting {len(documents)} documents")
        
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1}:")
            print(f"Metadata: {doc.metadata}")
            
            # Test document ID generation
            assert 'doc_id' in doc.metadata, "Document missing doc_id in metadata"
            assert len(doc.metadata['doc_id']) == 16, "Document ID not in expected format"
            
            # Test source file information
            assert 'source' in doc.metadata, "Document missing source in metadata"
            assert doc.metadata['source'].endswith('.pdf'), "Source file not a PDF"
            
    except Exception as e:
        pytest.fail(f"Document metadata test failed: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # Added -s to show print statements 