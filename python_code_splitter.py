from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

text = """student = {
    "name": "Alice",
    "age": 22,
    "major": "Computer Science"
}

# Accessing a value
print(f"Student Name: {student['name']}")

# Adding a new key-value pair
student["graduated"] = False"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0,
)

result = splitter.split_text(text)

print(len(result))
print(result)