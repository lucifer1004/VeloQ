//! Structural schema-contract assertions for NCU schema targets.
//!
//! These tests inspect the JSON Schema documents exposed by
//! `veloq ncu schema <target>`. They pin VeloQ's own contract
//! invariants: canonical list shape, row keys, tagged inspect rows,
//! and explicit outer discriminators for untagged row unions.

use anyhow::{Context, Result};
use serde_json::{Map, Value};

struct SchemaDoc {
    target: &'static str,
    root: Value,
    defs: Map<String, Value>,
}

impl SchemaDoc {
    fn for_target(target: &'static str) -> Result<Self> {
        let root = veloq_ncu::schema::schema_value_for(target)?;
        let defs = root
            .get("$defs")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        Ok(Self { target, root, defs })
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
        schema
            .get("$ref")
            .and_then(Value::as_str)
            .map(|_| self.resolve(schema))
            .and_then(|resolved| self.property(&resolved, name))
    }

    fn rows_item(&self) -> Result<Value> {
        let rows = self
            .property(&self.root, "rows")
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

fn enum_contains(schema: &Value, expected: &str) -> bool {
    schema
        .get("enum")
        .and_then(Value::as_array)
        .is_some_and(|values| values.iter().any(|value| value.as_str() == Some(expected)))
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

fn assert_row_schema_has_key(doc: &SchemaDoc, row_schema: &Value) -> Result<()> {
    let variants = row_schema
        .get("oneOf")
        .or_else(|| row_schema.get("anyOf"))
        .and_then(Value::as_array);
    if let Some(variants) = variants {
        for (index, variant) in variants.iter().enumerate() {
            assert_string_property(doc, variant, "key")
                .with_context(|| format!("{}: rows[] variant #{index}", doc.target))?;
        }
        return Ok(());
    }
    assert_string_property(doc, row_schema, "key")
}

fn assert_canonical_list_schema(doc: &SchemaDoc) -> Result<()> {
    if !allows_type(&doc.root, "object") {
        anyhow::bail!("{}: response schema root must be an object", doc.target);
    }
    for field in ["count", "total_matched", "rows"] {
        if !required_contains(&doc.root, field) {
            anyhow::bail!("{}: `{field}` must be required", doc.target);
        }
        let property = doc
            .property(&doc.root, field)
            .with_context(|| format!("{}: missing `{field}` property", doc.target))?;
        let expected_type = if field == "rows" { "array" } else { "integer" };
        if !allows_type(&property, expected_type) {
            anyhow::bail!(
                "{}: `{field}` should allow {expected_type}, got {property}",
                doc.target
            );
        }
    }
    assert_row_schema_has_key(doc, &doc.rows_item()?)
}

fn row_variants(row_schema: &Value) -> Result<Vec<Value>> {
    let raw = row_schema
        .get("oneOf")
        .or_else(|| row_schema.get("anyOf"))
        .and_then(Value::as_array)
        .context("row schema should expose union variants")?;
    Ok(raw.to_vec())
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

fn assert_required_field_in_some_variant(
    doc: &SchemaDoc,
    variants: &[Value],
    field: &str,
) -> Result<()> {
    for variant in variants {
        let resolved = doc.resolve(variant);
        if required_contains(&resolved, field) {
            return Ok(());
        }
    }
    anyhow::bail!("{}: no row variant requires `{field}`", doc.target)
}

fn assert_property_in_some_variant(doc: &SchemaDoc, variants: &[Value], field: &str) -> Result<()> {
    for variant in variants {
        let resolved = doc.resolve(variant);
        if doc.property(&resolved, field).is_some() {
            return Ok(());
        }
    }
    anyhow::bail!("{}: no row variant exposes `{field}`", doc.target)
}

fn assert_root_string_discriminator(doc: &SchemaDoc, field: &str) -> Result<()> {
    if !required_contains(&doc.root, field) {
        anyhow::bail!("{}: `{field}` discriminator must be required", doc.target);
    }
    assert_string_property(doc, &doc.root, field)
}

#[test]
fn every_schema_target_is_canonical_list_with_keyed_rows() -> Result<()> {
    for target in veloq_ncu::schema_targets::TARGETS {
        let doc = SchemaDoc::for_target(target.name)?;
        assert_canonical_list_schema(&doc)?;
    }
    Ok(())
}

#[test]
fn inspect_rows_expose_type_tagged_variants() -> Result<()> {
    let doc = SchemaDoc::for_target("inspect")?;
    let row = doc.rows_item()?;
    let values = tag_const_values(&row, "type")?;
    for expected in ["launch", "not_found"] {
        if !values.iter().any(|value| value == expected) {
            anyhow::bail!("inspect rows missing `{expected}` type variant");
        }
    }
    Ok(())
}

#[test]
fn metrics_rows_use_outer_format_discriminator_for_untagged_variants() -> Result<()> {
    let doc = SchemaDoc::for_target("metrics")?;
    if !required_contains(&doc.root, "format") {
        anyhow::bail!("metrics: `format` discriminator must be required");
    }
    let format = doc
        .property(&doc.root, "format")
        .context("metrics: missing `format` property")?;
    let resolved_format = doc.resolve(&format);
    for expected in ["long", "per_launch"] {
        if !enum_contains(&resolved_format, expected) {
            anyhow::bail!("metrics: format enum missing `{expected}`");
        }
    }

    let variants = row_variants(&doc.rows_item()?)?;
    assert_required_field_in_some_variant(&doc, &variants, "counter_name")?;
    assert_required_field_in_some_variant(&doc, &variants, "counters")?;
    Ok(())
}

#[test]
fn source_metric_and_warp_stall_rows_use_outer_axis_discriminator() -> Result<()> {
    let source_metrics = SchemaDoc::for_target("source-metrics")?;
    assert_root_string_discriminator(&source_metrics, "axis")?;
    let source_metric_variants = row_variants(&source_metrics.rows_item()?)?;
    for field in ["line", "address", "line_count"] {
        assert_required_field_in_some_variant(&source_metrics, &source_metric_variants, field)?;
    }

    let warp_stalls = SchemaDoc::for_target("warp-stalls")?;
    assert_root_string_discriminator(&warp_stalls, "axis")?;
    let warp_stall_variants = row_variants(&warp_stalls.rows_item()?)?;
    for field in ["line", "reason"] {
        assert_required_field_in_some_variant(&warp_stalls, &warp_stall_variants, field)?;
    }
    assert_property_in_some_variant(&warp_stalls, &warp_stall_variants, "rel_address")?;
    Ok(())
}
