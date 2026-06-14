import asyncio
import json
from pathlib import Path

from backend.embeddings.embeddings import embed_text
from backend.embeddings.retrieve_layers import build_catalog_embedding_text, select_catalog_columns
from backend.mcp.client.client import mcp_client
from backend.mcp.client.tools_discovery import get_database_catalog, get_table_schema

from config import DB_DESCRIPTIONS


def _infer_geometry_column(columns: list[dict[str, str]]) -> str | None:
    for col in columns:
        name = str(col.get("name") or "").strip()
        data_type = str(col.get("type") or "").strip().lower()
        if data_type in {"user-defined", "geometry"}:
            return name
        if name.lower() in {"geom", "geometry", "geometri"}:
            return name
    return None


async def build_metadata_catalog() -> list[dict]:
    catalog: list[dict] = []
    catalog_data = await get_database_catalog()

    for db_entry in catalog_data.get("databases", []):
        db_name = db_entry["database"]
        db_desc = DB_DESCRIPTIONS.get(db_name, "")

        for schema_entry in db_entry.get("schemas", []):
            schema = schema_entry["schema"]

            for table_entry in schema_entry.get("tables", []):
                table = table_entry["table"]
                schema_result = await get_table_schema(db_name, schema, table)
                raw_columns = schema_result.get("columns", []) if isinstance(schema_result, dict) else []
                columns = [
                    {
                        "name": str(col.get("column_name") or "").strip(),
                        "type": str(col.get("data_type") or "").strip(),
                    }
                    for col in raw_columns
                    if isinstance(col, dict) and col.get("column_name")
                ]
                geometry_column = (
                    table_entry.get("recommended_geometry_column")
                    or _infer_geometry_column(columns)
                )
                selected_columns = select_catalog_columns(
                    {"columns": columns, "geometry_column": geometry_column}
                )
                catalog.append(
                    {
                        "database": db_name,
                        "database_description": db_desc,
                        "schema": schema,
                        "table": table,
                        "full_name": f"{db_name}.{schema}.{table}",
                        "geometry_column": geometry_column,
                        "selected_columns": selected_columns,
                    }
                )

    return catalog


async def main() -> None:
    catalog = await build_metadata_catalog()
    index = []

    for item in catalog:
        #text = build_catalog_embedding_text(item)
        embedding_text = build_catalog_embedding_text(item)
        index.append(
            {
                **item,
                "embedding_text": embedding_text,
                "embedding": embed_text(embedding_text),
            }
        )

    output_path = Path(__file__).parent / "table_index.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    print(f"Index size: {len(index)}")
    print(f"Saved compact table index to: {output_path}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        asyncio.run(mcp_client.close())
