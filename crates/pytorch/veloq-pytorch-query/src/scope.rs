use schemars::JsonSchema;
use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize, JsonSchema)]
pub struct PytorchScope {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub step: Option<i64>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub aggregated_over: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RankScope {
    pub rank: Option<i64>,
    pub all_ranks: bool,
}

impl RankScope {
    pub fn echo(self, step: Option<i64>) -> PytorchScope {
        let mut aggregated_over = Vec::new();
        if self.all_ranks {
            aggregated_over.push("rank".to_string());
        }
        PytorchScope {
            rank: self.rank,
            worker: None,
            step,
            aggregated_over,
        }
    }
}
