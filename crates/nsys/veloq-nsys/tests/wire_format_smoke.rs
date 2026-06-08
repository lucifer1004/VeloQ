//! Structural schema-contract assertions for NSys schema targets.
//!
//! These tests inspect the JSON Schema documents exposed by
//! `veloq schema <target>`. They pin VeloQ's own contract invariants:
//! canonical list shape, row keys, tagged event rows, and the explicit
//! non-list exceptions. They deliberately do not validate fixture
//! payloads against schemas; that would mostly test serde/schemars
//! behavior.
//!
//! Boundary: the first tests below cover source-neutral list
//! invariants (`count`, `total_matched`, `rows`, row `key`). The
//! remaining tests are intentionally NSys-specific: hidden targets,
//! operational singleton payloads, source-tagged metrics bodies, and
//! NSys event-kind discriminators. The JSON Schema traversal helpers
//! stay test-local for now; cross-source extraction would obscure
//! those source-specific boundaries more than it would help.

use anyhow::{Context, Result};
use serde_json::{Map, Value};

const NON_CANONICAL_PUBLIC_TARGETS: &[&str] = &[
    "ncu-command",
    "metrics",
    "prep",
    "prep-status",
    "correlation-stats",
];

const SINGLETON_PUBLIC_TARGETS: &[&str] =
    &["ncu-command", "prep", "prep-status", "correlation-stats"];

struct SchemaDoc {
    target: &'static str,
    root: Value,
    defs: Map<String, Value>,
}

impl SchemaDoc {
    fn for_target(target: &'static str) -> Result<Self> {
        let root = veloq_nsys::schema::schema_value_for(target)?;
        Ok(Self::from_root(target, root))
    }

    fn from_root(target: &'static str, root: Value) -> Self {
        let defs = root
            .get("$defs")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        Self { target, root, defs }
    }

    fn resolve(&self, schema: &Value) -> Value {
        let Some(refstr) = schema.get("$ref").and_then(Value::as_str) else {
            return schema.clone();
        };
        let name = refstr.rsplit('/').next().unwrap_or(refstr);
        self.defs
            .get(name)
            .cloned()
            .unwrap_or_else(|| schema.clone())
    }

    fn property(&self, schema: &Value, name: &str) -> Option<Value> {
        if let Some(value) = schema
            .get("properties")
            .and_then(Value::as_object)
            .and_then(|properties| properties.get(name))
        {
            return Some(value.clone());
        }
        let refstr = schema.get("$ref").and_then(Value::as_str)?;
        let def_name = refstr.rsplit('/').next().unwrap_or(refstr);
        self.defs
            .get(def_name)
            .and_then(|resolved| self.property(resolved, name))
    }

    fn root_rows_item(&self) -> Result<Value> {
        self.rows_item_for(&self.root)
    }

    fn rows_item_for(&self, schema: &Value) -> Result<Value> {
        let rows = self
            .property(schema, "rows")
            .with_context(|| format!("{}: schema missing rows property", self.target))?;
        let items = rows
            .get("items")
            .with_context(|| format!("{}: rows.items missing", self.target))?;
        Ok(self.resolve(items))
    }
}

fn allows_type(schema: &Value, expected: &str) -> bool {
    match schema.get("type") {
        Some(Value::String(kind)) => kind == expected,
        Some(Value::Array(kinds)) => kinds.iter().any(|kind| kind.as_str() == Some(expected)),
        _ => false,
    }
}

fn required_contains(schema: &Value, field: &str) -> bool {
    schema
        .get("required")
        .and_then(Value::as_array)
        .is_some_and(|required| required.iter().any(|item| item.as_str() == Some(field)))
}

fn assert_string_property(doc: &SchemaDoc, schema: &Value, field: &str) -> Result<()> {
    let property = doc
        .property(schema, field)
        .with_context(|| format!("{}: schema missing `{field}` property", doc.target))?;
    if !allows_type(&property, "string") {
        anyhow::bail!(
            "{}: `{field}` property should allow string, got {property}",
            doc.target
        );
    }
    Ok(())
}

fn union_variants(row_schema: &Value) -> Result<Vec<Value>> {
    let raw = row_schema
        .get("oneOf")
        .or_else(|| row_schema.get("anyOf"))
        .and_then(Value::as_array)
        .context("row schema should expose union variants")?;
    Ok(raw.to_vec())
}

fn assert_row_schema_has_key(doc: &SchemaDoc, row_schema: &Value) -> Result<()> {
    if let Ok(variants) = union_variants(row_schema) {
        for (index, variant) in variants.iter().enumerate() {
            assert_string_property(doc, variant, "key")
                .with_context(|| format!("{}: rows[] variant #{index}", doc.target))?;
        }
        return Ok(());
    }
    assert_string_property(doc, row_schema, "key")
}

fn assert_canonical_list_shape(doc: &SchemaDoc, schema: &Value) -> Result<()> {
    let schema = doc.resolve(schema);
    if !allows_type(&schema, "object") {
        anyhow::bail!("{}: response schema root must be an object", doc.target);
    }
    for field in ["count", "total_matched", "rows"] {
        if !required_contains(&schema, field) {
            anyhow::bail!("{}: `{field}` must be required", doc.target);
        }
        let property = doc
            .property(&schema, field)
            .with_context(|| format!("{}: missing `{field}` property", doc.target))?;
        let expected_type = if field == "rows" { "array" } else { "integer" };
        if !allows_type(&property, expected_type) {
            anyhow::bail!(
                "{}: `{field}` should allow {expected_type}, got {property}",
                doc.target
            );
        }
    }
    assert_row_schema_has_key(doc, &doc.rows_item_for(&schema)?)
}

fn tag_const_values(row_schema: &Value, tag: &str) -> Result<Vec<String>> {
    let variants = row_schema
        .get("oneOf")
        .and_then(Value::as_array)
        .context("tagged rows should use oneOf variants")?;
    let mut values = Vec::new();
    for (index, variant) in variants.iter().enumerate() {
        let value = variant
            .get("properties")
            .and_then(Value::as_object)
            .and_then(|properties| properties.get(tag))
            .and_then(|schema| schema.get("const"))
            .and_then(Value::as_str)
            .with_context(|| format!("variant #{index} missing `{tag}` const"))?;
        values.push(value.to_string());
    }
    Ok(values)
}

fn assert_tag_values_include(values: &[String], expected: &[&str], label: &str) -> Result<()> {
    for expected_value in expected {
        if !values.iter().any(|value| value == expected_value) {
            anyhow::bail!("{label} missing `{expected_value}` variant");
        }
    }
    Ok(())
}

fn assert_required_field_in_every_variant(
    doc: &SchemaDoc,
    row_schema: &Value,
    field: &str,
) -> Result<()> {
    for (index, variant) in union_variants(row_schema)?.iter().enumerate() {
        let resolved = doc.resolve(variant);
        if !required_contains(&resolved, field) {
            anyhow::bail!(
                "{}: rows[] variant #{index} should require `{field}`",
                doc.target
            );
        }
    }
    Ok(())
}

fn target_is_allowlisted(target: &str) -> bool {
    NON_CANONICAL_PUBLIC_TARGETS.contains(&target)
}

#[test]
fn every_public_schema_target_is_canonical_list_or_explicit_exception() -> Result<()> {
    for target in veloq_nsys::schema_targets::TARGETS {
        if target_is_allowlisted(target.name) {
            continue;
        }
        let doc = SchemaDoc::for_target(target.name)?;
        assert_canonical_list_shape(&doc, &doc.root)?;
    }
    Ok(())
}

#[test]
fn non_canonical_public_targets_are_explicit_and_present() -> Result<()> {
    for allowlisted in NON_CANONICAL_PUBLIC_TARGETS {
        if !veloq_nsys::schema_targets::TARGETS
            .iter()
            .any(|target| target.name == *allowlisted)
        {
            anyhow::bail!("non-canonical target `{allowlisted}` is not public");
        }
    }
    Ok(())
}

#[test]
fn singleton_public_targets_do_not_claim_primary_rows() -> Result<()> {
    for target in SINGLETON_PUBLIC_TARGETS {
        let doc = SchemaDoc::for_target(target)?;
        if !allows_type(&doc.root, "object") {
            anyhow::bail!("{target}: singleton target root must be an object");
        }
        if doc.property(&doc.root, "rows").is_some() {
            anyhow::bail!("{target}: singleton target must not expose `rows`");
        }
    }
    Ok(())
}

#[test]
fn metrics_schema_variants_are_canonical_lists_with_keyed_rows() -> Result<()> {
    let doc = SchemaDoc::for_target("metrics")?;
    let variants = doc
        .root
        .get("oneOf")
        .and_then(Value::as_array)
        .context("metrics root should expose source-tagged variants")?;
    let mut sources = Vec::new();
    for (index, variant) in variants.iter().enumerate() {
        let source = variant
            .get("properties")
            .and_then(Value::as_object)
            .and_then(|properties| properties.get("source"))
            .and_then(|schema| schema.get("const"))
            .and_then(Value::as_str)
            .with_context(|| format!("metrics variant #{index} missing `source` const"))?;
        sources.push(source.to_string());
        assert_canonical_list_shape(&doc, variant)
            .with_context(|| format!("metrics source `{source}`"))?;
    }
    assert_tag_values_include(
        &sources,
        &["gpu", "nic", "cpu-sampling", "cpu-sched"],
        "metrics sources",
    )
}

#[test]
fn hidden_schema_targets_are_canonical_lists_with_keyed_rows() -> Result<()> {
    for target in veloq_nsys::schema_targets::HIDDEN_TARGETS {
        let root = (target.schema_fn)()?;
        let doc = SchemaDoc::from_root(target.name, root);
        assert_canonical_list_shape(&doc, &doc.root)?;
    }
    Ok(())
}

#[test]
fn search_event_ref_schema_exposes_type_tagged_variants() -> Result<()> {
    let doc = SchemaDoc::for_target("search")?;
    let row = doc.root_rows_item()?;
    let values = tag_const_values(&row, "type")?;
    assert_tag_values_include(
        &values,
        &[
            "kernel",
            "memcpy",
            "memset",
            "nvtx",
            "sync",
            "runtime",
            "osrt",
            "graph",
            "graph_node",
            "graph_event",
            "cuda_event",
            "overhead",
        ],
        "search EventRef",
    )?;
    for field in ["key", "row_id", "name", "start_ns", "duration_ns"] {
        assert_required_field_in_every_variant(&doc, &row, field)?;
    }
    Ok(())
}

#[test]
fn inspect_rows_expose_type_tagged_variants() -> Result<()> {
    let doc = SchemaDoc::for_target("inspect")?;
    let row = doc.root_rows_item()?;
    let values = tag_const_values(&row, "type")?;
    assert_tag_values_include(
        &values,
        &[
            "kernel",
            "memcpy",
            "memset",
            "runtime",
            "osrt",
            "nvtx",
            "sync",
            "graph",
            "graph_node",
            "graph_event",
            "cuda_event",
            "overhead",
            "cpu_sample",
            "not_found",
        ],
        "inspect EventDetails",
    )
}

#[test]
fn correlate_embedded_events_use_event_ref_schema() -> Result<()> {
    let doc = SchemaDoc::for_target("correlate")?;
    let row = doc.root_rows_item()?;
    let events = doc
        .property(&row, "events")
        .context("correlate rows should expose `events`")?;
    let event_item = events
        .get("items")
        .map(|items| doc.resolve(items))
        .context("correlate events should expose array items")?;
    let values = tag_const_values(&event_item, "type")?;
    assert_tag_values_include(
        &values,
        &["kernel", "memcpy", "memset", "runtime", "sync", "graph"],
        "correlate EventRef",
    )?;
    assert_row_schema_has_key(&doc, &event_item)
}

#[test]
fn slices_rows_expose_instance_and_aggregate_variants() -> Result<()> {
    let doc = SchemaDoc::for_target("slices")?;
    let row = doc.root_rows_item()?;
    let variants = union_variants(&row)?;
    if variants.len() != 2 {
        anyhow::bail!("slices rows should expose exactly two variants");
    }
    for field in ["key", "name"] {
        assert_required_field_in_every_variant(&doc, &row, field)?;
    }
    Ok(())
}
