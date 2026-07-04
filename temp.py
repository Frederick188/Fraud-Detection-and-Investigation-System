from src.rag import retrieve

docs = retrieve("What is merchant category code?")

for d in docs:
    print("="*50)
    print(d.page_content)