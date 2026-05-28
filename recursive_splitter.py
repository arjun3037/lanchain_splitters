from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader(file_path="./assets/dl-curriculum.pdf")
documents = loader.load()



splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    # it return the token length of the text, which is important for language models that have token limits,
    #  if we use character length then splitter split based on the character and 1 character is not equals to 1 token,
    #  so we need to use token length to split the text properly.
    length_function=lambda x: len(tiktoken.get_encoding("cl100k_base").encode(x)),
)


docs = splitter.split_documents(documents)

for i , doc in enumerate(docs):
    doc.metadata["source"] = f"page_{i+1}"
    doc.metadata["page"] = i+1
    doc.metadata["total_pages"] = len(docs)
    doc.metadata["chunk_number"] = i


for doc in docs:
    print(doc.metadata)
    print("--"*30)
    print(doc.page_content )
    print("="*60)


