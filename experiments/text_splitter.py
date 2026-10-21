from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Docker is a containerization platform that allows developers to package
applications together with their dependencies.

Docker images are templates used to create containers. Images contain
application code, libraries, dependencies, and configuration.

Docker containers are running instances of Docker images. Containers
provide isolated and consistent environments.

Docker Compose allows developers to define and run applications containing
multiple containers.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))

for index, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {index + 1} ---")
    print(chunk)
