import chromadb
import ollama
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
import time
import shutil
import tempfile
import json
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field, model_validator
from contextlib import asynccontextmanager

#Models configuration
DEFAULT_LANGUAGE_MODELS = {
    "en": {
        "embedding": "nomic-embed-text:137m-v1.5-fp16",
        "llm": "llama3:8b",
        "prompt_template": """Based ONLY on the following context, answer the question.
Context language: '{language}'. Question language: '{language}'.
Filter used for context: {query_meta_dict}.
If the answer cannot be found in the context, state that and why (e.g., "Based on the provided documents, the answer is not available.").
Do not use external knowledge. Do not mention source filenames unless explicitly part of the content.

Context:
{context_string}

Question:
{query}

Answer ({language}):"""
    },
    "it": {
        "embedding": "granite-embedding:278m",
        "llm": "mistral-nemo:12b",
        "prompt_template": """Basandoti ESCLUSIVAMENTE sul seguente contesto, rispondi alla domanda.
Lingua del contesto: '{language}'. Lingua della domanda: '{language}'.
Filtro utilizzato per il contesto: {query_meta_dict}.
Se la risposta non può essere trovata nel contesto, indicalo e spiega il motivo (ad es. "Sulla base dei documenti forniti, la risposta non è disponibile.").
Non utilizzare conoscenze esterne. Non menzionare i nomi dei file sorgente a meno che non facciano esplicitamente parte del contenuto.

Contesto:
{context_string}

Domanda:
{query}

Risposta ({language}):"""
    }
}

#FastAPI Setup
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./metaRAG_db")
OLLAMA_API_HOST = os.getenv("OLLAMA_API_HOST", None)
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "metaRAG_collection")

#Pydantic models
class PydanticModel_BaseDocument_Metadata(BaseModel):
    user_id: str = Field(..., description="The ID of the user uploading or querying the document.")
    language: str = Field(..., description="The language code (e.g., 'en', 'it') of the document content.")
    model_config = {
        "extra": "allow"
    }

class PydanticModel_TextUpload_Request(BaseModel):
    text_content: str = Field(..., description="The plain text content to upload.")
    document_metadata: PydanticModel_BaseDocument_Metadata = Field(...,
                                                    description="Metadata associated with the text, must include user_id and language.",
                                                    examples=[{
                                                        "user_id": "test_user_id",
                                                        "language": "it",
                                                        "topic": "test topic value"
                                                        }]
                                                    )
    source_identifier: str = Field("direct_text_upload", description="A string to identify the source of this text, used as 'source_filename' in metadata.")

class PydanticModel_Ask_Request(BaseModel):
    query: str = Field(..., description="The question to ask the RAG system.")
    query_metadata_filter: PydanticModel_BaseDocument_Metadata = Field(...,
                                                        description="Metadata associated with the text, must include user_id and language.",
                                                        examples=[{
                                                            "user_id": "test_user_id",
                                                            "language": "it",
                                                            "topic": "test topic value"
                                                            }]
                                                        )
    n_results_for_context: int = Field(3, description="Number of context chunks to retrieve.", gt=0)

class PydanticModel_Api_Response(BaseModel):
    status: str = "success"
    message: Optional[str] = None
    data: Optional[Any] = None

class PydanticModel_ContextItem(BaseModel):
    text: str = Field(..., description="The text content of the retrieved context chunk.")
    metadata: Dict[str, Any] = Field(...,
                                     description="The full metadata associated with this context chunk. Complex values are deserialized from JSON strings if possible.",
                                      examples=[{
                                        "user_id": "test_user_id",
                                        "language": "it",
                                        "source_filename": "file_name_test.txt",
                                        "topic": "test topic value"
                                        }]
                                )
    distance: Optional[float] = Field(None, description="The distance score of this chunk from the query embedding (if available).")

class PydanticModel_AskResponse_Data(BaseModel):
    answer: str = Field(..., description="The answer generated by the LLM.")
    retrieved_context_items: List[PydanticModel_ContextItem] = Field(..., description="A list of context items used to generate the answer.")

class PydanticModel_Ask_Api_Response(PydanticModel_Api_Response):
    data: Optional[PydanticModel_AskResponse_Data] = None

class metaRAG:
    _MANDATORY_METADATA_KEYS = {"user_id", "language"}

    def __init__(
        self,
        chroma_path: str = "./metaRAG_db",
        collection_name: str = "metaRAG_collection",
        language_models: Dict[str, Dict[str, str]] = DEFAULT_LANGUAGE_MODELS,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        ollama_host: Optional[str] = None
    ):
        print("[metaRAG INFO] Initializing RAG System for API Server...")
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.language_models = language_models
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_host = ollama_host

        if not self.language_models:
            raise ValueError("`language_models` configuration cannot be empty.")
        print(f"[metaRAG INFO] Configured languages: {', '.join(self.language_models.keys())}")

        try:
            self.ollama_client = ollama.Client(host=self.ollama_host)
            self.ollama_client.list()
            print("[metaRAG INFO] Successfully connected to Ollama.")
            self._verify_language_configurations()
        except Exception as e:
            print(f"[metaRAG ERROR] Failed to connect to Ollama or verify configurations: {e}")
            raise ConnectionError(f"Could not connect to Ollama service or required models/configurations missing: {e}") from e

        try:
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"[metaRAG INFO] ChromaDB collection '{self.collection_name}' loaded/created at '{self.chroma_path}'. Count: {self.collection.count()}")
        except Exception as e:
            print(f"[metaRAG ERROR] Failed to initialize ChromaDB: {e}")
            raise ConnectionError(f"Could not initialize ChromaDB: {e}") from e

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        print("[metaRAG INFO] RAG System Initialized Successfully for API Server.")

    def _validate_and_extract_mandatory_metadata(self, metadata_dict: Dict[str, Any], context_msg: str) -> Tuple[str, str]:
        if isinstance(metadata_dict, BaseModel):
            meta_dict_internal = metadata_dict.model_dump(exclude_unset=True)
        else:
            meta_dict_internal = metadata_dict

        missing_keys = self._MANDATORY_METADATA_KEYS - set(meta_dict_internal.keys())
        if missing_keys:
            raise ValueError(f"{context_msg}: Missing mandatory metadata keys: {missing_keys}. Must include 'user_id' and 'language'.")

        user_id = meta_dict_internal["user_id"]
        language = meta_dict_internal["language"]

        if not isinstance(user_id, str) or not user_id:
            raise ValueError(f"{context_msg}: 'user_id' in metadata must be a non-empty string.")
        if not isinstance(language, str) or not language:
            raise ValueError(f"{context_msg}: 'language' in metadata must be a non-empty string.")

        if language not in self.language_models:
            raise ValueError(f"{context_msg}: Language '{language}' from metadata is not configured. Configured: {list(self.language_models.keys())}")

        return str(user_id), str(language)

    def _verify_language_configurations(self):
        print("[metaRAG INFO] Verifying language configurations and Ollama model availability...")
        try:
            available_models_info = self.ollama_client.list()['models']
            available_models = {m['model'] for m in available_models_info}
        except Exception as e:
             print(f"[metaRAG ERROR] Could not list models from Ollama: {e}")
             raise ConnectionError("Failed to verify models with Ollama.") from e

        required_ollama_models = set()
        for lang_code, config in self.language_models.items():
            if "embedding" not in config or not isinstance(config["embedding"], str) or not config["embedding"]:
                 raise ValueError(f"Language configuration for '{lang_code}' is missing, not a string, or has an empty 'embedding' key.")
            if "llm" not in config or not isinstance(config["llm"], str) or not config["llm"]:
                 raise ValueError(f"Language configuration for '{lang_code}' is missing, not a string, or has an empty 'llm' key.")
            if "prompt_template" not in config or not isinstance(config["prompt_template"], str) or not config["prompt_template"]:
                 raise ValueError(f"Language configuration for '{lang_code}' is missing, not a string, or has an empty 'prompt_template' key.")

            required_ollama_models.add(config['embedding'])
            required_ollama_models.add(config['llm'])

        missing_ollama_models = required_ollama_models - available_models
        if missing_ollama_models:
            print(f"[metaRAG ERROR] Missing required Ollama models: {', '.join(missing_ollama_models)}")
            raise ValueError(f"Missing required Ollama models: {missing_ollama_models}")
        print("[metaRAG INFO] All configured Ollama models are available and language configurations are complete.")

    def _get_models_and_prompt_for_language(self, lang: str) -> Tuple[str, str, str]:
        config = self.language_models.get(lang)
        if not config:
            raise ValueError(f"Unsupported language specified: '{lang}'. Configured languages are: {list(self.language_models.keys())}")

        embedding_model = config.get("embedding")
        llm_model = config.get("llm")
        prompt_template = config.get("prompt_template")

        if not embedding_model or not llm_model or not prompt_template:
            missing_keys = []
            if not embedding_model: missing_keys.append("embedding")
            if not llm_model: missing_keys.append("llm")
            if not prompt_template: missing_keys.append("prompt_template")
            raise ValueError(f"Language configuration for '{lang}' is incomplete. Missing or empty keys: {missing_keys}.")

        return embedding_model, llm_model, prompt_template


    def _get_ollama_embedding(self, text: str, language: str, user_id_for_log: str) -> List[float]:
        try:
            embedding_model, _, _ = self._get_models_and_prompt_for_language(language)
            response = self.ollama_client.embeddings(model=embedding_model, prompt=text)
            return response["embedding"]
        except ValueError as ve:
             print(f"[User: {user_id_for_log}] metaRAG ERROR: Cannot get embedding model for language '{language}': {ve}")
             raise
        except Exception as e:
            embedding_model_name = "unknown"
            try:
                model_conf = self.language_models.get(language, {})
                embedding_model_name = model_conf.get("embedding", "unknown_config_for_lang")
            except Exception: pass
            
            print(f"[User: {user_id_for_log}] metaRAG ERROR: Embedding failed with model '{embedding_model_name}' (lang: {language}): {e}")
            raise RuntimeError(f"Ollama embedding failed for model {embedding_model_name} (lang: {language})") from e

    def _load_and_split_document_from_file(self, file_path: str, user_id_for_log: str, original_filename: str) -> Tuple[List[str], str]:
        path = Path(file_path)
        file_name_for_meta = original_filename

        if not path.exists():
            print(f"[User: {user_id_for_log}] metaRAG ERROR: Temp file not found: {file_path}")
            return [], file_name_for_meta

        text_content = ""
        try:
            file_suffix = Path(original_filename).suffix.lower()
            if file_suffix == ".pdf":
                reader = PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text(); text_content += page_text + "\n" if page_text else ""
                print(f"[User: {user_id_for_log}] metaRAG INFO: Extracted text from PDF: {original_filename}")
            elif file_suffix == ".txt":
                with open(file_path, 'r', encoding='utf-8') as f: text_content = f.read()
                print(f"[User: {user_id_for_log}] metaRAG INFO: Read text from TXT: {original_filename}")
            else:
                print(f"[User: {user_id_for_log}] metaRAG WARNING: Unsupported file type: {file_suffix} for {original_filename}. Skipping.")
                return [], file_name_for_meta

            if not text_content.strip():
                print(f"[User: {user_id_for_log}] metaRAG WARNING: No text extracted from: {original_filename}")
                return [], file_name_for_meta

            chunks = self.text_splitter.split_text(text_content)
            print(f"[User: {user_id_for_log}] metaRAG INFO: Split doc '{original_filename}' into {len(chunks)} chunks.")
            return chunks, file_name_for_meta
        except Exception as e:
            print(f"[User: {user_id_for_log}] metaRAG ERROR: Error processing file {original_filename}: {e}")
            return [], file_name_for_meta

    #Serializes complex metadata values (dicts, lists) into JSON strings. Keeps scalar values (str, int, float, bool) as they are.
    def _serialize_metadata_values(self, metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        serialized_meta = {}
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                try:
                    serialized_meta[key] = json.dumps(value)
                except TypeError as e:
                    print(f"[metaRAG WARNING] Could not serialize metadata value for key '{key}': {value}. Error: {e}. Storing as string representation.")
                    serialized_meta[key] = str(value)
            elif isinstance(value, (str, int, float, bool)):
                serialized_meta[key] = value
            elif value is None:
                print(f"[metaRAG WARNING] Metadata key '{key}' has None value. Skipping or converting to empty string.")
                serialized_meta[key] = ""
            else:
                print(f"[metaRAG WARNING] Metadata key '{key}' has unsupported type {type(value)}. Converting to string: {value}")
                serialized_meta[key] = str(value)
        return serialized_meta

    def _process_and_store_chunks(
        self,
        chunks: List[str],
        document_metadata_dict: Dict[str, Any],
        source_name_for_metadata: str,
        user_id: str,
        language: str
    ):
        if not chunks:
            print(f"[User: {user_id}] metaRAG WARNING: No chunks provided for '{source_name_for_metadata}'. Skipping storage.")
            return

        try:
            embedding_model, _, _ = self._get_models_and_prompt_for_language(language)
        except ValueError as e:
            print(f"[User: {user_id}] metaRAG ERROR: Cannot proceed with storing chunks for '{source_name_for_metadata}'. {e}")
            raise

        print(f"[User: {user_id}] metaRAG INFO: Generating embeddings for {len(chunks)} chunks from '{source_name_for_metadata}' using '{embedding_model}' (Lang: {language})...")

        embeddings = []
        valid_chunks = []
        chunk_ids = []
        final_metadatas_for_chunks = []

        base_chunk_meta_pre_serialization = document_metadata_dict.copy()
        base_chunk_meta_pre_serialization["source_filename"] = source_name_for_metadata
        serialized_base_chunk_meta = self._serialize_metadata_values(base_chunk_meta_pre_serialization)

        for i, chunk_text in enumerate(chunks):
            try:
                embedding = self._get_ollama_embedding(chunk_text, language=language, user_id_for_log=user_id)
                embeddings.append(embedding)
                valid_chunks.append(chunk_text)
                chunk_id = f"{user_id}_{source_name_for_metadata.replace(os.sep, '_').replace(' ', '_')}_{i}_{language}_{time.time_ns()}"
                chunk_ids.append(chunk_id)
                final_metadatas_for_chunks.append(serialized_base_chunk_meta.copy())
            except Exception as e:
                print(f"[User: {user_id}] metaRAG ERROR: Skipping chunk {i} from '{source_name_for_metadata}' due to error: {e}")

        if not valid_chunks:
            print(f"[User: {user_id}] metaRAG ERROR: No valid embeddings generated for '{source_name_for_metadata}'. Document/text not added.")
            return

        try:
            if final_metadatas_for_chunks:
                print(f"[User: {user_id}] metaRAG DEBUG: Example chunk metadata being added for '{source_name_for_metadata}': {final_metadatas_for_chunks[0]}")
            print(f"[User: {user_id}] metaRAG INFO: Adding {len(valid_chunks)} chunks to ChromaDB (Lang: {language}, Source: '{source_name_for_metadata}')...")
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=valid_chunks,
                metadatas=final_metadatas_for_chunks
            )
            print(f"[User: {user_id}] metaRAG INFO: Successfully added {len(valid_chunks)} chunks from '{source_name_for_metadata}'. Collection total count: {self.collection.count()}")
        except Exception as e:
            if final_metadatas_for_chunks:
                print(f"[User: {user_id}] metaRAG DEBUG: Metadata that might have caused error: {final_metadatas_for_chunks[0] if final_metadatas_for_chunks else 'N/A'}")
            print(f"[User: {user_id}] metaRAG ERROR: Failed to add chunks to ChromaDB for '{source_name_for_metadata}': {e}")
            raise RuntimeError(f"Failed to add chunks to ChromaDB for '{source_name_for_metadata}'") from e

    def upload_document(
        self,
        temp_file_path: str,
        original_filename: str,
        document_metadata: Union[PydanticModel_BaseDocument_Metadata, Dict[str, Any]]
    ):
        if isinstance(document_metadata, BaseModel):
            doc_meta_dict = document_metadata.model_dump(exclude_unset=True)
        else:
            doc_meta_dict = document_metadata

        try:
            user_id, language = self._validate_and_extract_mandatory_metadata(
                doc_meta_dict, f"Upload of file '{original_filename}'"
            )
        except ValueError as e:
            raise

        print(f"[User: {user_id}] metaRAG INFO: Starting file upload processing: {original_filename} (Lang from meta: {language})")
        if "source_filename" in doc_meta_dict:
            print(f"[User: {user_id}] metaRAG WARNING: 'source_filename' in provided metadata for file upload ('{original_filename}') will be overwritten by the actual filename.")

        chunks, actual_source_name = self._load_and_split_document_from_file(temp_file_path, user_id, original_filename)
        self._process_and_store_chunks(chunks, doc_meta_dict, actual_source_name, user_id, language)
        return actual_source_name

    def upload_text(
        self,
        text_content: str,
        document_metadata: Union[PydanticModel_BaseDocument_Metadata, Dict[str, Any]],
        source_identifier: str = "direct_text_upload"
    ):
        if isinstance(document_metadata, BaseModel):
            doc_meta_dict = document_metadata.model_dump(exclude_unset=True)
        else:
            doc_meta_dict = document_metadata

        try:
            user_id, language = self._validate_and_extract_mandatory_metadata(
                doc_meta_dict, f"Upload of text (source: '{source_identifier}')"
            )
        except ValueError as e:
            raise

        print(f"[User: {user_id}] metaRAG INFO: Starting text upload (Source ID: '{source_identifier}', Lang from meta: {language})")
        if "source_filename" in doc_meta_dict:
             print(f"[User: {user_id}] metaRAG WARNING: 'source_filename' in provided metadata for text upload will be overwritten by the 'source_identifier' parameter ('{source_identifier}').")

        if not text_content or not text_content.strip():
            raise ValueError("text_content cannot be empty.")

        try:
            chunks = self.text_splitter.split_text(text_content)
            print(f"[User: {user_id}] metaRAG INFO: Split text '{source_identifier}' into {len(chunks)} chunks.")
        except Exception as e:
            raise RuntimeError(f"Error splitting text content for source '{source_identifier}'") from e

        self._process_and_store_chunks(chunks, doc_meta_dict, source_identifier, user_id, language)
        return source_identifier

    def _retrieve_context(
        self,
        query: str,
        query_metadata_filter_dict: Dict[str, Any],
        n_results: int = 3
    ) -> List[PydanticModel_ContextItem]:
        user_id, language = self._validate_and_extract_mandatory_metadata(
            query_metadata_filter_dict, "Context Retrieval"
        )

        print(f"[User: {user_id}] metaRAG INFO: Retrieving context for query (Lang from filter: {language}): '{query[:50]}...'")
        print(f"[User: {user_id}] metaRAG INFO: Applying metadata filter: {query_metadata_filter_dict}")

        retrieved_items: List[PydanticModel_ContextItem] = []
        try:
            query_embedding = self._get_ollama_embedding(query, language=language, user_id_for_log=user_id)
            where_conditions = []
            for key, value in query_metadata_filter_dict.items():
                # ChromaDB metadata values cannot be None (they are not stored).
                if value is not None:
                    where_conditions.append({key: value})
                else:
                    print(f"[User: {user_id}] metaRAG DEBUG: Skipping metadata filter for key '{key}' because its value is None.")


            if not where_conditions:
                print(f"[User: {user_id}] metaRAG WARNING: No valid (non-None) filter conditions found in query_metadata_filter_dict. Querying without a 'where' clause. This might be unintended.")
                final_where_filter = None #Query all documents
            elif len(where_conditions) == 1:
                final_where_filter = where_conditions[0]
            else:
                final_where_filter = {"$and": where_conditions}

            print(f"[User: {user_id}] metaRAG DEBUG: ChromaDB query filter being used: {final_where_filter}")

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=final_where_filter,
                include=['documents', 'metadatas', 'distances']
            )

            retrieved_docs = results.get('documents', [[]])[0]
            retrieved_metadatas = results.get('metadatas', [[]])[0]
            retrieved_distances = results.get('distances', [[]])[0]

            if retrieved_docs and retrieved_metadatas and retrieved_distances and len(retrieved_docs) == len(retrieved_metadatas) == len(retrieved_distances):
                 print(f"[User: {user_id}] metaRAG INFO: Retrieved {len(retrieved_docs)} context chunks matching filters.")
                 for doc, meta, dist in zip(retrieved_docs, retrieved_metadatas, retrieved_distances):
                    retrieved_items.append(PydanticModel_ContextItem(text=doc, metadata=meta, distance=dist))
            elif not retrieved_docs:
                 print(f"[User: {user_id}] metaRAG WARNING: No relevant documents found matching all filter criteria: {query_metadata_filter_dict}")
            else:
                print(f"[User: {user_id}] metaRAG WARNING: Mismatch in lengths of retrieved documents, metadatas, or distances from ChromaDB.")
                for i in range(len(retrieved_docs)):
                    doc = retrieved_docs[i]
                    meta = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {"error": "missing metadata"}
                    dist = retrieved_distances[i] if i < len(retrieved_distances) else None
                    retrieved_items.append(PydanticModel_ContextItem(text=doc, metadata=meta, distance=dist))
            return retrieved_items
        except ValueError as ve:
             print(f"[User: {user_id}] metaRAG ERROR: Cannot retrieve context: {ve}")
             raise
        except Exception as e:
            print(f"[User: {user_id}] metaRAG ERROR: Error retrieving context from ChromaDB: {e}", exc_info=True)
            raise RuntimeError(f"Error retrieving context from ChromaDB: {e}")

    def ask(
        self,
        query: str,
        query_metadata_filter: Union[PydanticModel_BaseDocument_Metadata, Dict[str, Any]],
        n_results_for_context: int = 3
        ) -> Tuple[str, List[PydanticModel_ContextItem]]:
        if isinstance(query_metadata_filter, BaseModel):
            query_meta_dict = query_metadata_filter.model_dump(exclude_unset=True)
        else:
            query_meta_dict = query_metadata_filter

        user_id, language = self._validate_and_extract_mandatory_metadata(
            query_meta_dict, "Ask"
        )

        print(f"\n[User: {user_id}] metaRAG INFO: Processing 'ask' request (Lang from filter: {language})...")

        retrieved_context_items_list: List[PydanticModel_ContextItem] = self._retrieve_context(
            query,
            query_metadata_filter_dict=query_meta_dict,
            n_results=n_results_for_context
        )

        context_string = f"No relevant context found matching the filter: {query_meta_dict}."
        if retrieved_context_items_list:
            context_texts = [item.text for item in retrieved_context_items_list]
            context_string = "\n\n---\n\n".join(context_texts)
            print(f"[User: {user_id}] metaRAG INFO: Context prepared for LLM (Lang: {language}).")
        else:
             print(f"[User: {user_id}] metaRAG WARNING: Proceeding to LLM without retrieved context.")

        try:
            _, llm_model, language_specific_prompt_template_str = self._get_models_and_prompt_for_language(language)
        except ValueError as e:
            print(f"[User: {user_id}] metaRAG ERROR: Configuration error for language '{language}': {e}")
            raise # Re-raise to be caught by endpoint

        print(f"[User: {user_id}] metaRAG INFO: Using LLM '{llm_model}' and prompt template for language '{language}'.")

        try:
            final_prompt = language_specific_prompt_template_str.format(
                language=language,
                query_meta_dict=str(query_meta_dict),
                context_string=context_string,
                query=query
            )
        except KeyError as ke:
            print(f"[User: {user_id}] metaRAG ERROR: Invalid prompt template for language '{language}'. Missing key: {ke}")
            raise RuntimeError(f"Invalid prompt template for language '{language}'. Misconfigured placeholder: {ke}") from ke


        try:
            start_time = time.time()
            response = self.ollama_client.chat(
                model=llm_model,
                messages=[{'role': 'user', 'content': final_prompt}],
            )
            end_time = time.time()
            final_answer = response['message']['content']
            print(f"[User: {user_id}] metaRAG INFO: LLM response in {end_time - start_time:.2f}s.")
            return final_answer.strip(), retrieved_context_items_list
        except Exception as e:
            print(f"[User: {user_id}] metaRAG ERROR: LLM inference error with '{llm_model}' (lang: {language}): {e}")
            raise RuntimeError(f"LLM inference error with '{llm_model}' (lang: {language})") from e

#global rag_system_instance: to be initialized later in 'lifespan' definition
rag_system_instance: Optional[metaRAG] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_system_instance
    print("[LIFESPAN STARTUP] Initializing metaRAG system...")
    try:
        rag_system_instance = metaRAG(
            chroma_path=CHROMA_DB_PATH,
            collection_name=RAG_COLLECTION_NAME,
            ollama_host=OLLAMA_API_HOST,
            # language_models can be overridden here if needed, otherwise uses DEFAULT_LANGUAGE_MODELS
        )
        print("[LIFESPAN STARTUP] metaRAG system initialized successfully.")
    except (ConnectionError, ValueError) as e:
        print(f"[LIFESPAN STARTUP FATAL] Could not initialize metaRAG: {e}")
        rag_system_instance = None

    if rag_system_instance is None:
        print("[LIFESPAN STARTUP WARNING] metaRAG instance is None. API will not function correctly.")
    else:
        print("[LIFESPAN STARTUP] FastAPI application startup: metaRAG instance is ready.")

    yield

    print("[LIFESPAN SHUTDOWN] FastAPI application shutting down.")
    #From here you can delete temporary files, delete unnecessary data, or clean ChromaDB if it is being tested.


app = FastAPI(
    title="metaRAG API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health/", response_model=PydanticModel_Api_Response, tags=["Utility"])
async def health_check():
    if rag_system_instance is None:
        raise HTTPException(status_code=503, detail="metaRAG failed to initialize or is not available.")
    try:
        rag_system_instance.ollama_client.list()
        rag_system_instance.collection.count()
        return PydanticModel_Api_Response(message="metaRAG API is healthy.")
    except Exception as e:
        print(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {e}")

@app.post("/upload/file/", response_model=PydanticModel_Api_Response, tags=["Upload"])
async def upload_file_endpoint(
    document_file: UploadFile = File(..., description="The document file (PDF or TXT) to upload."),
    metadata_json: str = Form(..., description="A JSON string representing the document's metadata. Must include 'user_id' and 'language', plus any custom fields.")
):
    if rag_system_instance is None:
        raise HTTPException(status_code=503, detail="metaRAG not available. Service may not have started correctly.")

    try:
        metadata_dict = json.loads(metadata_json)
        doc_metadata = PydanticModel_BaseDocument_Metadata(**metadata_dict)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format for metadata_json.")
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid metadata structure: {e}")

    tmp_file_path = None
    try:
        suffix = Path(document_file.filename).suffix if document_file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(document_file.file, tmp)
            tmp_file_path = tmp.name

        original_filename = document_file.filename if document_file.filename else "uploaded_file"

        source_name = rag_system_instance.upload_document(
            temp_file_path=tmp_file_path,
            original_filename=original_filename,
            document_metadata=doc_metadata
        )
        return PydanticModel_Api_Response(message=f"File '{source_name}' uploaded and processed successfully.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        print(f"Error during file upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during file upload: {e}")
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        if document_file:
            await document_file.close()


@app.post("/upload/text/", response_model=PydanticModel_Api_Response, tags=["Upload"])
async def upload_text_endpoint(request: PydanticModel_TextUpload_Request):
    if rag_system_instance is None:
        raise HTTPException(status_code=503, detail="metaRAG not available. Service may not have started correctly.")
    try:
        source_name = rag_system_instance.upload_text(
            text_content=request.text_content,
            document_metadata=request.document_metadata,
            source_identifier=request.source_identifier
        )
        return PydanticModel_Api_Response(message=f"Text content from source '{source_name}' uploaded and processed successfully.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        print(f"Error during text upload: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during text upload: {e}")

@app.post("/ask/", response_model=PydanticModel_Ask_Api_Response, tags=["Query"])
async def ask_endpoint(request: PydanticModel_Ask_Request):
    #Asks a question to the RAG system. The response includes the answer and detailed information about the context items retrieved from ChromaDB.

    if rag_system_instance is None:
        raise HTTPException(status_code=503, detail="metaRAG not available. Service may not have started correctly.")
    try:
        answer, context_items_list = rag_system_instance.ask(
            query=request.query,
            query_metadata_filter=request.query_metadata_filter,
            n_results_for_context=request.n_results_for_context
        )

        return PydanticModel_Ask_Api_Response(
            message="Query processed successfully.",
            data=PydanticModel_AskResponse_Data(
                answer=answer,
                retrieved_context_items=context_items_list
            )
        )
    except ValueError as ve: #Catches validation errors
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re: #Catches operational errors (Ollama, ChromaDB, prompt template issues)
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        print(f"Error during ask: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing your question: {e}")

#Start Uvicorn server programmatically
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
