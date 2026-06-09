use crate::PytorchCommandResult;

#[derive(serde::Serialize)]
pub struct SchemaPayload {
    pub target: String,
    pub schema: serde_json::Value,
}

pub fn schema_value_for(target: &str) -> PytorchCommandResult<serde_json::Value> {
    crate::schema_targets::resolve(target)
}
