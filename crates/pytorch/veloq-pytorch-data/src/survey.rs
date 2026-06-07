use crate::model::{Event, TraceFileSchemaSurvey, TraceSchemaSurvey, TypedArgCoverage};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

const MISSING_FIELD: &str = "(missing)";
const NON_OBJECT_EVENT: &str = "(non-object)";
const NON_STRING_FIELD: &str = "(non-string)";

#[derive(Debug, Default)]
pub(crate) struct TraceSchemaSurveyBuilder {
    survey: TraceSchemaSurvey,
}

impl TraceSchemaSurveyBuilder {
    pub(crate) fn record_file_header(&mut self, trace_index: u32, top: &Map<String, Value>) {
        let mut top_level_keys = top
            .keys()
            .filter(|key| key.as_str() != "traceEvents")
            .cloned()
            .collect::<Vec<_>>();
        top_level_keys.sort();
        self.survey.files.push(TraceFileSchemaSurvey {
            trace_index,
            top_level_keys,
            has_device_properties: top.contains_key("deviceProperties"),
            ..TraceFileSchemaSurvey::default()
        });
    }

    pub(crate) fn record_raw_trace_event(
        &mut self,
        trace_index: u32,
        raw_obj: Option<&Map<String, Value>>,
    ) {
        self.survey.raw_event_count += 1;
        if let Some(file) = self.file_mut(trace_index) {
            file.raw_event_count += 1;
        }

        let Some(obj) = raw_obj else {
            bump(&mut self.survey.phase_counts, NON_OBJECT_EVENT);
            bump(&mut self.survey.category_counts, NON_OBJECT_EVENT);
            return;
        };

        bump(&mut self.survey.phase_counts, field_label(obj, "ph"));
        bump(&mut self.survey.category_counts, field_label(obj, "cat"));
        if let Some(args) = obj.get("args").and_then(Value::as_object) {
            for key in args.keys() {
                bump(&mut self.survey.arg_key_counts, key);
            }
        }
    }

    pub(crate) fn record_flow_marker(&mut self, trace_index: u32) {
        self.survey.flow_marker_count += 1;
        if let Some(file) = self.file_mut(trace_index) {
            file.flow_marker_count += 1;
        }
    }

    pub(crate) fn record_parsed_event(&mut self, event: &Event) {
        self.survey.parsed_event_count += 1;
        bump(
            &mut self.survey.event_type_counts,
            event.event_type.as_str(),
        );
        self.survey.typed_arg_coverage.record_event(event);
        if let Some(file) = self.file_mut(event.trace_index) {
            file.parsed_event_count += 1;
        }
    }

    pub(crate) fn record_skipped_event(&mut self, trace_index: u32) {
        self.survey.skipped_event_count += 1;
        if let Some(file) = self.file_mut(trace_index) {
            file.skipped_event_count += 1;
        }
    }

    pub(crate) fn finish(self) -> TraceSchemaSurvey {
        self.survey
    }

    fn file_mut(&mut self, trace_index: u32) -> Option<&mut TraceFileSchemaSurvey> {
        self.survey
            .files
            .iter_mut()
            .find(|file| file.trace_index == trace_index)
    }
}

impl TypedArgCoverage {
    fn record_event(&mut self, event: &Event) {
        if event.rank.is_some() {
            self.rank += 1;
        }
        if event.worker.is_some() {
            self.worker += 1;
        }
        if event.device_id.is_some() {
            self.device_id += 1;
        }
        if event.stream_id.is_some() {
            self.stream_id += 1;
        }
        if event.external_id.is_some() {
            self.external_id += 1;
        }
        if event.correlation_id.is_some() {
            self.correlation_id += 1;
        }
        if event.step.is_some() {
            self.step += 1;
        }
        if event.bytes.is_some() {
            self.bytes += 1;
        }
        if event.shape.is_some() {
            self.shape += 1;
        }
    }
}

fn field_label(obj: &Map<String, Value>, field: &str) -> String {
    let Some(value) = obj.get(field) else {
        return MISSING_FIELD.to_string();
    };
    if let Some(value) = value.as_str() {
        return value.to_string();
    }
    NON_STRING_FIELD.to_string()
}

fn bump(map: &mut BTreeMap<String, usize>, key: impl AsRef<str>) {
    let count = map.entry(key.as_ref().to_string()).or_insert(0);
    *count += 1;
}
