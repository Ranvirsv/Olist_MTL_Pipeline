import duckdb
from pydantic import ValidationError
from src.schemas import OlistAnalyticalRow
from tqdm import tqdm
from loguru import logger

def build_data_warehouse(root_dir: str):
    duckdb_olist_conn = duckdb.connect(f"{root_dir}/data/olist.duckdb")
    logger.info("Connected to database")

    with open(f"{root_dir}/sql/build_db.sql", "r") as f:
        build_db_file = f.read()

    duckdb_olist_conn.execute(build_db_file)

    logger.success("Database built successfully")

    query = "SELECT * FROM analytical_base_table;"

    olist_df = duckdb_olist_conn.execute(query).fetchdf()
    records = olist_df.to_dict(orient='records')

    validated_data = []
    val_errors = 0

    for i, record in tqdm(enumerate(records[:10000])):
        try:
            validated_row = OlistAnalyticalRow(**record)
            validated_data.append(validated_row)
        except ValidationError as e:
            logger.error(f"\nValidation failed at index: {i}")
            val_errors += 1


    if val_errors > 0:
        logger.error(f"\nPipeline Failed: Found {val_errors} schema violations.!")
    else:
        logger.success("\nData Contract Passed! Database is perfectly structured!")

