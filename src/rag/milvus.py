from pymilvus import DataType, MilvusClient

from .paths import DATA_DIR

DB_PATH = DATA_DIR / "milvus.db"
COLLECTION_NAME = "devops_documents"
EMBEDDING_DIM = 1024

DATA_DIR.mkdir(parents=True, exist_ok=True)
client = MilvusClient(str(DB_PATH))

schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=500)
schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=500)
schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=10000)
schema.add_field(
    field_name="embedding",
    datatype=DataType.FLOAT_VECTOR,
    dim=EMBEDDING_DIM,
)

index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="AUTOINDEX",
    metric_type="COSINE",
)


def ensure_collection() -> bool:
    """Create the collection and vector index if they do not already exist."""
    if not client.has_collection(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            schema=schema,
            index_params=index_params,
        )
        return True

    if not client.list_indexes(COLLECTION_NAME, field_name="embedding"):
        client.create_index(COLLECTION_NAME, index_params)
    client.load_collection(COLLECTION_NAME)

    return False


if __name__ == "__main__":
    try:
        created = ensure_collection()
        status = "créée" if created else "déjà présente"
        print(f"Collection '{COLLECTION_NAME}' {status} et indexée.")
        print("Collections disponibles :", client.list_collections())
    finally:
        client.close()
