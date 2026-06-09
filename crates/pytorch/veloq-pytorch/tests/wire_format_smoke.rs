//! Structural schema-contract assertions for PyTorch schema targets.
//!
//! These tests inspect the JSON Schema documents exposed by
//! `veloq pytorch schema <target>`. They pin VeloQ's own contract
//! invariants: canonical list shape, row keys, and agent-visible
//! discriminators. They deliberately do not validate fixture payloads
//! against schemas; that would mostly test serde/schemars behavior.
//!
//! Boundary: the canonical list check is source-neutral. The remaining
//! tests are PyTorch-specific: `search` keeps an open string event
//! `type`, while `slices` exposes source-specific mode variants. The
//! JSON Schema traversal helpers stay test-local until a common
//! abstraction can preserve those boundaries without hiding them.

use anyhow::{Context, Result};
use serde_json::{Map, Value};

struct SchemaDoc {
    target: &'static str,
    root: Value,
    defs: Map<String, Value>,
}

impl SchemaDoc {
    fn for_target(target: &'static str) -> Result<Self> {
        let root = veloq_pytorch::schema::schema_value_for(target)?;
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

#[test]
fn every_schema_target_is_canonical_list_with_keyed_rows() -> Result<()> {
    for target in veloq_pytorch::schema_targets::TARGETS {
        let doc = SchemaDoc::for_target(target.name)?;
        assert_canonical_list_schema(&doc)?;
    }
    Ok(())
}

#[test]
fn event_ref_schema_exposes_open_type_discriminator() -> Result<()> {
    let doc = SchemaDoc::for_target("search")?;
    let row = doc.rows_item()?;
    for field in ["key", "row_id", "type", "name", "start_ns", "duration_ns"] {
        if !required_contains(&row, field) {
            anyhow::bail!("search EventRef must require `{field}`");
        }
    }
    assert_string_property(&doc, &row, "type")?;
    Ok(())
}

#[test]
fn slices_rows_expose_mode_tagged_variants() -> Result<()> {
    let doc = SchemaDoc::for_target("slices")?;
    let row = doc.rows_item()?;
    let values = tag_const_values(&row, "mode")?;
    assert_tag_values_include(&values, &["instance", "aggregate"], "slices rows")
}
