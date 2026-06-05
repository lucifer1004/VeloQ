use anyhow::Result;
use veloq_pytorch_query::{
    CollectivesResponse, CorrelateResponse, InspectResponse, PrepResponse, SearchResponse,
    SlicesResponse, StatsResponse, SummaryResponse, TimelineResponse,
};

#[derive(serde::Serialize)]
pub struct SchemaPayload {
    pub target: String,
    pub schema: serde_json::Value,
}

pub fn schema_value_for(target: &str) -> Result<serde_json::Value> {
    let value = match target {
        "summary" => serde_json::to_value(schemars::schema_for!(SummaryResponse))?,
        "search" => serde_json::to_value(schemars::schema_for!(SearchResponse))?,
        "inspect" => serde_json::to_value(schemars::schema_for!(InspectResponse))?,
        "stats" => serde_json::to_value(schemars::schema_for!(StatsResponse))?,
        "correlate" => serde_json::to_value(schemars::schema_for!(CorrelateResponse))?,
        "timeline" => serde_json::to_value(schemars::schema_for!(TimelineResponse))?,
        "slices" => serde_json::to_value(schemars::schema_for!(SlicesResponse))?,
        "collectives" => serde_json::to_value(schemars::schema_for!(CollectivesResponse))?,
        "prep" => serde_json::to_value(schemars::schema_for!(PrepResponse))?,
        other => anyhow::bail!(
            "unknown pytorch schema target `{other}`; expected one of: summary, search, inspect, stats, correlate, timeline, slices, collectives, prep"
        ),
    };
    Ok(value)
}
