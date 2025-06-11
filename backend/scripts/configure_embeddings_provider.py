import os
import sys
from sqlalchemy import create_engine, Column, Integer, String, Boolean, LargeBinary, select, update, insert, inspect
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from sqlalchemy.exc import IntegrityError
import urllib.parse

from dotenv import load_dotenv
load_dotenv('../../.env')

DB_USER = "postgres"
_DB_PASSWORD_RAW = "password"
DB_HOST = "relational_db"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_PASSWORD = urllib.parse.quote_plus(_DB_PASSWORD_RAW)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL_NAME = "text-embedding-3-small"
OPENAI_MODEL_DIM = 1536

class EmbeddingProviderEnum:
    OPENAI = "OPENAI"

class IndexModelStatusEnum:
    PRESENT = "PRESENT"
    PAST = "PAST"

Base = declarative_base()

class CloudEmbeddingProvider(Base):
    __tablename__ = "embedding_provider"
    provider_type = Column(String(50), primary_key=True, nullable=False) # provider_type IS the PK
    api_key = Column(LargeBinary, nullable=True)
    api_url = Column(String, nullable=True)        # Added based on your \d output
    api_version = Column(String, nullable=True)    # Added based on your \d output
    deployment_name = Column(String, nullable=True)# Added based on your \d output
    # NO 'id' column here

class SearchSettings(Base):
    __tablename__ = "search_settings"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    model_dim = Column(Integer, nullable=False)
    normalize = Column(Boolean, nullable=False)
    query_prefix = Column(String, nullable=False, server_default="")
    passage_prefix = Column(String, nullable=False, server_default="")
    status = Column(String, nullable=False)
    index_name = Column(String, nullable=False, unique=True)
    provider_type = Column(String)
    multipass_indexing = Column(Boolean, nullable=False, server_default="false")
    num_rerank = Column(Integer, nullable=False, server_default="10")

def get_db_session(db_user, db_password, db_host, db_port, db_name):
    db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    try:
        with engine.connect() as connection:
            pass
    except Exception as e:
        print(f"Failed to connect to the database: {e}", file=sys.stderr)
        sys.exit(1)
    return SessionLocal()

def clean_model_name_for_index(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_")

def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    db: Session = get_db_session(DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME)

    try:
        provider = db.execute(
            select(CloudEmbeddingProvider).where(CloudEmbeddingProvider.provider_type == EmbeddingProviderEnum.OPENAI)
        ).scalar_one_or_none()

        if provider:
            db.execute(
                update(CloudEmbeddingProvider)
                .where(CloudEmbeddingProvider.provider_type == EmbeddingProviderEnum.OPENAI)
                .values(api_key=OPENAI_API_KEY.encode('utf-8'))
            )
        else:
            new_provider = CloudEmbeddingProvider(
                provider_type=EmbeddingProviderEnum.OPENAI,
                api_key=OPENAI_API_KEY.encode('utf-8')
            )
            db.add(new_provider)
        db.flush()

        current_present_settings = db.execute(
            select(SearchSettings).where(SearchSettings.status == IndexModelStatusEnum.PRESENT)
        ).scalars().all()

        for setting in current_present_settings:
            if setting.model_name == OPENAI_MODEL_NAME and setting.provider_type == EmbeddingProviderEnum.OPENAI:
                setting.model_dim = OPENAI_MODEL_DIM
                setting.normalize = True
                setting.query_prefix = ""
                setting.passage_prefix = ""
                db.add(setting)
                db.commit()
                print(f"'{OPENAI_MODEL_NAME}' is already the PRESENT model and has been re-verified/updated.")
                return
            else:
                setting.status = IndexModelStatusEnum.PAST
                db.add(setting)
        db.flush()
        
        new_index_name = f"danswer_chunk_{clean_model_name_for_index(OPENAI_MODEL_NAME)}"
        
        existing_openai_setting_by_index_name = db.execute(
             select(SearchSettings).where(SearchSettings.index_name == new_index_name)
        ).scalar_one_or_none()

        default_search_settings_values = {
            "model_name": OPENAI_MODEL_NAME,
            "model_dim": OPENAI_MODEL_DIM,
            "normalize": True,
            "query_prefix": "",
            "passage_prefix": "",
            "status": IndexModelStatusEnum.PRESENT,
            "index_name": new_index_name,
            "provider_type": EmbeddingProviderEnum.OPENAI,
            "multipass_indexing": False,
            "num_rerank": 10,
        }

        if existing_openai_setting_by_index_name:
            db.execute(
                update(SearchSettings)
                .where(SearchSettings.index_name == new_index_name)
                .values(**default_search_settings_values)
            )
        else:
            new_search_setting_stmt = insert(SearchSettings).values(**default_search_settings_values)
            db.execute(new_search_setting_stmt)

        db.commit()
        print(f"Database configuration updated. '{OPENAI_MODEL_NAME}' is now the PRESENT embedding model.")

    except IntegrityError as e:
        db.rollback()
        print(f"An IntegrityError occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        db.rollback()
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    for model_cls in [CloudEmbeddingProvider, SearchSettings]:
        if not inspect(model_cls).primary_key:
            print(f"Model {model_cls.__name__} does not have a primary key.", file=sys.stderr)
            sys.exit(1)
    main()