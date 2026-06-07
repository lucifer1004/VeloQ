use crate::{PytorchCommandError, PytorchCommandResult};
use veloq_pytorch_query::{
    CollectivesResponse, CorrelateResponse, InspectResponse, PrepResponse, SearchResponse,
    SlicesResponse, StatsResponse, SummaryResponse, TimelineResponse,
};

#[derive(serde::Serialize)]
pub struct SchemaPayload {
    pub target: String,
    pub schema: serde_json::Value,
}

pub fn schema_value_for(target: &str) -> PytorchCommandResult<serde_json::Value> {
    let schema = match target {
        "summary" => schemars::schema_for!(SummaryResponse),
        "search" => schemars::schema_for!(SearchResponse),
        "inspect" => schemars::schema_for!(InspectResponse),
        "stats" => schemars::schema_for!(StatsResponse),
        "correlate" => schemars::schema_for!(CorrelateResponse),
        "timeline" => schemars::schema_for!(TimelineResponse),
        "slices" => schemars::schema_for!(SlicesResponse),
        "collectives" => schemars::schema_for!(CollectivesResponse),
        "prep" => schemars::schema_for!(PrepResponse),
        other => return Err(PytorchCommandError::unknown_schema_target(other)),
    };
    serde_json::to_value(schema)
        .map_err(|source| PytorchCommandError::serialize_schema(target, source))
}
