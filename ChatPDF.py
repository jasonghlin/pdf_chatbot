import os
import io
import streamlit as st
import pdfplumber
import pandas as pd
import torch
import numpy as np
import torch
from PIL import Image
from uuid import uuid4
from transformers import CLIPProcessor, CLIPModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from langchain.text_splitter import TokenTextSplitter
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class CLIPTextEmbeddings:
    def __init__(self, clip_model, clip_processor):
        self.clip_model = clip_model
        self.clip_processor = clip_processor

    def embed_query(self, query: str):

        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.squeeze(0).cpu().numpy()



class RAGPDFChatbot:
    def __init__(self, 
                 embedding_model_name='blevlabs/stella_en_v5', 
                 ollama_model='llama3', 
                 qdrant_host='qdrant', 
                 qdrant_port=6333):
        """
        初始化 RAG 系統，支持多模態嵌入
        """

        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


        self.qdrant_client = QdrantClient(
            url=qdrant_host,  
            port=qdrant_port
        )   


        self.ollama_llm = OllamaLLM(model=ollama_model, base_url="http://localhost:11434")
        

        self.vector_store_text = None
        self.vector_store_images = None
        self.rag_chain_text = None
        self.rag_chain_images = None
        
        self.retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        self.pdf_processed = False
        self.debug_mode = False
        self.clip_text_embeddings = CLIPTextEmbeddings(self.clip_model, self.clip_processor)

    def create_qdrant_collection(self, collection_name, vector_size=1024):
        """
        創建 Qdrant 集合，支援不同的嵌入維度
        """
        try:
            existing_collections = self.qdrant_client.get_collections().collections
            if not any(col.name == collection_name for col in existing_collections):
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                st.success(f"Created Qdrant collection: {collection_name}")
            else:
                st.info(f"Collection {collection_name} already exists")
        except Exception as e:
            st.error(f"Error creating Qdrant collection: {e}")
    
    def text_embedding(self, text):
        """
        生成文本嵌入
        """
        embedding = self.embedding_model.embed_query(str(text))
        embedding = np.array(embedding)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def image_embedding(self, image_path):
        """
        生成圖像嵌入使用CLIP
        """
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                image_features = image_features / np.linalg.norm(image_features)

            return image_features.squeeze(0).cpu().numpy()
        except Exception as e:
            st.error(f"Error embedding image {image_path}: {e}")
            return None

    def process_pdf(self, pdf_docs, collection_name="pdf_content"):
        """
        處理 PDF 文件並存儲到 Qdrant，支援圖文類型
        """
        try:
            
            self.create_qdrant_collection(f"{collection_name}_text", vector_size=1024)
            self.create_qdrant_collection(f"{collection_name}_images", vector_size=512)
            

            raw_content = self._extract_pdf_content(pdf_docs)
            


            text_splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=100)
            text_chunks = text_splitter.split_text(raw_content['text'])
            
            # debug mode
            st.write("### Debug: Extracted Content")
            st.write(f"Text chunks: {len(text_chunks)}")
            st.write(f"Tables: {len(raw_content['tables'])}")
            st.write(f"Images: {len(raw_content['images'])}")
            st.write(f"Vector Graphics: {len(raw_content['vector_graphics'])}")

            text_points = []
            for chunk in text_chunks:
                embedding = self.text_embedding(chunk)
                point=models.PointStruct(
                        id=str(uuid4()),
                        payload={
                            "page_content": chunk,
                            "type": "text",
                            "metadata": {  
                                "type": "text"  
                            }
                        },
                        vector=embedding.tolist(),
                    )
                text_points.append(point)
            
            if text_points:
                self.qdrant_client.upsert(
                    collection_name=f"{collection_name}_text",
                    points=text_points
                )
            
            table_points = []
            for table in raw_content.get('tables', []):
                embedding = self.text_embedding(table)
                
                point=models.PointStruct(
                        id=str(uuid4()),
                        payload={
                            "page_content": table,
                            "type": "table",
                            "metadata": {  
                                "type": "table"  
                            }
                        },
                        vector=embedding.tolist(),
                    )
                
                table_points.append(point)
            
            if table_points:
                self.qdrant_client.upsert(
                    collection_name=f"{collection_name}_text",
                    points=table_points
                )
            
            
            image_points = []
            for img_path in raw_content.get('images', []):
                embedding = self.image_embedding(img_path)
                if embedding is not None:
                    point=models.PointStruct(
                            id=str(uuid4()),
                            payload={
                                "page_content": img_path,
                                "path": img_path,
                                "type": "image",
                                "metadata": {  
                                "type": "image"  
                            }
                            },
                            vector=embedding.tolist(),
                        )
                    image_points.append(point)
            
            for vec_img_path in raw_content.get('vector_graphics', []):
                embedding = self.image_embedding(vec_img_path)
                if embedding is not None:
                    point=models.PointStruct(
                            id=str(uuid4()),
                            payload={
                                "page_content": vec_img_path, 
                                "path": vec_img_path,
                                "type": "vector_graphics",
                                "metadata": {  
                                "type": "vector_graphics"  
                            }
                            },
                            vector=embedding.tolist(),
                        )


                    image_points.append(point)

            if image_points:
                self.qdrant_client.upsert(
                    collection_name=f"{collection_name}_images",
                    points=image_points
                )

            # print("### Debug: First text point payload")
            # if text_points:
            #     first_point = text_points[0]
            #     print(first_point.payload)
            
            self.vector_store_text = Qdrant(
                client=self.qdrant_client, 
                collection_name=f"{collection_name}_text",
                embeddings=self.embedding_model,
                metadata_payload_key="metadata"
            )
            
            self.vector_store_images = Qdrant(
                client=self.qdrant_client, 
                collection_name=f"{collection_name}_images",
                embeddings=self.clip_text_embeddings.embed_query,
                metadata_payload_key="metadata"
            )


            retriever_text = self.vector_store_text.as_retriever(
                search_kwargs={
                    "k": 15,  
                    "filter": None,  
                    "score_threshold": 0.2,  
                }
            )

            retriever_images = self.vector_store_images.as_retriever(
                search_kwargs={
                    "k": 5,  
                    "filter": None,  
                    "score_threshold": 0.2,  
                }
            )
            
            # debug mode
            if self.debug_mode:
                st.write("### Debug: Vector Store Information")
                st.write(f"Total documents in text collection: {self.qdrant_client.count(collection_name=f'{collection_name}_text')}")
                st.write(f"Total documents in image collection: {self.qdrant_client.count(collection_name=f'{collection_name}_images')}")
            
            combine_docs_chain = create_stuff_documents_chain(
                self.ollama_llm, 
                self.retrieval_qa_chat_prompt
            )
            
            self.rag_chain_text = create_retrieval_chain(
                retriever_text, 
                combine_docs_chain
            )

            self.rag_chain_images = create_retrieval_chain(
                retriever_images, 
                combine_docs_chain
            )
            
            
            if (self.vector_store_text or self.vector_store_images) and (self.rag_chain_text or self.rag_chain_images):
                st.success("PDF 處理成功！")
                self.pdf_processed = True
                return raw_content
            else:
                st.error("創建向量存儲或 RAG 鏈失敗")
                return None
    
        except Exception as e:
            st.error(f"處理 PDF 時發生錯誤: {e}")
            return None
    
    
    def retrieval_qa(self, query, collection_name):
        """
        使用檢索增強生成回答問題，並包含不同類型的上下文
        """
        if not self.pdf_processed:
            st.error("請先處理 PDF 文件")
            return None
        
        text_retriever = Qdrant(
        client=self.qdrant_client, 
        collection_name=f"{collection_name}_text",
        embeddings=self.embedding_model
    ).as_retriever(
        search_kwargs={
            "k": 15,  
            "score_threshold": 0.2,
        }
    )

        clip_text_embeddings = CLIPTextEmbeddings(self.clip_model, self.clip_processor)

        image_retriever = Qdrant(
            client=self.qdrant_client, 
            collection_name=f"{collection_name}_images",
            embeddings=clip_text_embeddings.embed_query
        ).as_retriever(
            search_kwargs={
                "k": 5,  
                "score_threshold": 0.2,
            }
        )
    

        text_docs = text_retriever.invoke(query)
        image_docs = image_retriever.invoke(query)
        
        # 合併文檔
        retrieved_docs = text_docs + image_docs

        # print("retrieved_docs: ", retrieved_docs)
        
        # debug 調試輸出
        # st.write("### Debug: Retrieved Documents Details")
        # for i, doc in enumerate(retrieved_docs, 1):
        #     st.write(f"Document {i} Full Metadata: {doc.metadata}")
        #     st.write(f"Document {i}:")
        #     st.write(f"Metadata: {doc.metadata}")
        #     st.write(f"Content Type: {doc.metadata.get('type', 'Unknown')}")
        #     st.write(f"Content Preview: {doc.page_content[:200]}")
        #     st.write("---")

        # if self.debug_mode:
        #     st.write("### Debug: Retrieved Documents")
        #     st.write(f"Total retrieved documents: {len(retrieved_docs)}")
            
        #     # 顯示每個檢索到的文檔的詳細信息
        #     for i, doc in enumerate(retrieved_docs, 1):
        #         st.write(f"Document {i}:")
        #         st.write(f"Type: {doc.metadata.get('type', 'Unknown')}")
        #         st.write(f"Content Preview: {doc.page_content[:200]}...")
        #         st.write("---")

        result_text = self.rag_chain_text.invoke({"input": query})
        result_images = self.rag_chain_images.invoke({"input": query})
        
        results = {"result_text":result_text, "result_images": result_images}

        enhanced_context = []
        for doc in retrieved_docs:
            context_item = {
                'type': doc.metadata.get('type') or doc.metadata.get('metadata', {}).get('type', 'unknown'),
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            enhanced_context.append(context_item)
        
        results['retrieved_docs'] = enhanced_context
        
        return results
    
    def _extract_pdf_content(self, pdf_docs):
        """
        提取 PDF 文件的全部內容，包括文本、表格、圖像、LaTeX
        """
        all_texts = []
        all_tables = []
        all_images = []

        all_vector_graphics = []

        for pdf in pdf_docs:
            with open(pdf.name, "wb") as f:
                f.write(pdf.read())
            

            with pdfplumber.open(pdf.name) as doc:
                for page in doc.pages:
                    text = page.extract_text()
                    if text:
                        all_texts.append(text)
                    
                    tables = page.find_tables()
                    for table in tables:
                        table_data = table.extract()
                        if table_data:
                            table_str = "\n".join([", ".join(map(str, row)) for row in table_data])
                            all_tables.append(table_str)
            
            try:
                with pdfplumber.open(pdf.name) as doc:
                    for page_num, page in enumerate(doc.pages):
                        for img_index, img in enumerate(page.images):
                            img_data = doc.extract_image(img['object_id'])
                            img_bytes = img_data['image']
                            img_ext = img_data['ext']
                            img_filename = f"images/page{page_num}_img{img_index + 1}.{img_ext}"
                            os.makedirs("images", exist_ok=True)
                            with open(img_filename, "wb") as f:
                                f.write(img_bytes)
                            all_images.append(img_filename)
            except Exception as e:
                st.error(f"提取圖像時出錯: {e}")
            
            try:
                import fitz
                doc = fitz.open(pdf.name)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    vector_filename = f"vector_graphics/page_{page_num + 1}_vector.png"
                    os.makedirs("vector_graphics", exist_ok=True)
                    pix.save(vector_filename)
                    all_vector_graphics.append(vector_filename)
                doc.close()
            except Exception as e:
                st.error(f"提取 vector graph 時出錯: {e}")

        return {
            'text': ' '.join(all_texts),
            'tables': all_tables,
            'images': all_images,
            'vector_graphics': all_vector_graphics,
        }

def main():
    st.set_page_config(page_title="Ollama RAG PDF Chat", page_icon=":books:")
    st.header("Chat with PDFs using Llama3 🤖📚")
    
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGPDFChatbot()
    
    # PDF 上傳與處理
    with st.sidebar:
        st.subheader("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDFs here", accept_multiple_files=True)
        
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                if pdf_docs:
                    st.session_state.rag_system.process_pdf(pdf_docs)
    
    # 問題輸入與回答
    question = st.text_input("Ask a question about your documents")
    
    if question:
        with st.spinner("Generating answer..."):
            results = st.session_state.rag_system.retrieval_qa(question, collection_name="pdf_content")
            if results:
                st.write("### Answer")
                for key in results:
                   if key != "retrieved_docs":
                       result = results[key]
                       st.write(result['answer'])
                
                st.write("### Source Documents")
                for key in results:
                   result = results[key]
                   if key != "retrieved_docs": 
                        st.text_area("Content", result["answer"], height=100, key=str(uuid4()))
                   if key == "retrieved_docs":
                        st.write("### Retrieved Documents")
                        for doc in result:
                            st.write(f"### {doc['type'].capitalize()} Content")
                            st.write(f"Type: {doc['type']}")
                            st.write(f"Metadata: {doc.get('metadata', {})}")
                            st.text_area("Content", doc['content'], height=100, key=str(uuid4()))
                            print("doc type before judge: ", doc["type"])
                            
                            if (doc['type'] == 'image' or doc['type'] == 'vector_graphics'):
                                print("doc type: ", doc["type"])
                                st.image(doc['content'])
                            elif doc['type'] == 'table':
                                try:
                                    
                                    df = pd.read_csv(io.StringIO(doc['content']))
                                    st.table(df)
                                except Exception as e:
                                    st.error(f"Table parsing error: {e}")   

if __name__ == "__main__":
    main()