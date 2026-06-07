use crate::dto::{PrepAuxiliary, PrepResponse, PrepRow, TraceSchemaSurveyDto};
use veloq_pytorch_data::PrepState;

pub fn prep_response(state: PrepState, built: bool) -> PrepResponse {
    let rows = state
        .sidecars
        .iter()
        .map(|sidecar| PrepRow {
            key: sidecar.key.clone(),
            name: sidecar.name.clone(),
            path: sidecar.path.clone(),
            present: sidecar.present,
        })
        .collect::<Vec<_>>();
    PrepResponse {
        count: rows.len(),
        total_matched: rows.len(),
        rows,
        auxiliary: PrepAuxiliary {
            input_path: state.input_path,
            artifact_dir: state.artifact_dir,
            cache_version: state.cache_version,
            cache_fresh: state.cache_fresh,
            built,
            schema_survey: state.schema_survey.as_ref().map(TraceSchemaSurveyDto::from),
        },
    }
}
