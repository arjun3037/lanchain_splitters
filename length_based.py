from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader



loader = PyPDFLoader(file_path='./assets/dl-curriculum.pdf')
documents = loader.load()


splitter = CharacterTextSplitter(
    chunk_size=100, 
    chunk_overlap=20 , 
    separator=''
)

result = splitter.split_documents(documents)
print(result[0].page_content)
print('------------------')
print(result[1].page_content)

