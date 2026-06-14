

from dataclasses import dataclass
from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict

from backend.database import PoolManager

@dataclass
class AppContext:
    db: PoolManager
    conn_ids: dict[str, str]
    step_results: dict[str, dict[int, dict]]
    duckdb_path: str




# --------------------------------------------------
# Plan model
# --------------------------------------------------

class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    action: Literal[
        "query", "filter", "select_columns", "spatial_join", "attribute_join",
        "buffer", "nearest", "distance", "aggregate", "sort", "limit",
        "clip", "union", "intersection", "difference", "dissolve", "centroid", "transform_crs",
    ]
    database: Optional[str] = None
    sql: Optional[str] = None
    validSql: Optional[str] = None


    source_step: Optional[int] = None
    left_source: Optional[int] = None
    right_source: Optional[int] = None


    join_type: Optional[Literal["intersects", "within", "contains", "dwithin", "touches", "overlaps"]] = None
    left_geom: Optional[str] = None
    right_geom: Optional[str] = None


    distance_meters: Optional[float] = None
    buffer_distance: Optional[float] = None


    filters: list[str] = Field(default_factory=list)   

    select_columns: list[str] = Field(default_factory=list)


    group_by: list[str] = Field(default_factory=list)


    aggregation_type: Optional[Literal["count", "sum", "avg", "min", "max"]] = None


    metric_column: Optional[str] = None
    metric_as: Optional[str] = None


    order_by: Optional[str] = None
    order_direction: Optional[Literal["asc", "desc"]] = None


    limit: Optional[int] = None


    target_crs: Optional[str] = None


    join_keys: list[str] = Field(default_factory=list)  
                  
    schema_name: Optional[str] = None
    table_name: Optional[str] = None


class QueryPlan(BaseModel):
    query_mode: Literal["single_db", "multi_db"]
    steps: list[PlanStep] = Field(default_factory=list)
