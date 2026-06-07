use duckdb::types::Value;

/// SQL text plus the positional parameters introduced by that text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SqlFragment {
    pub sql: String,
    pub params: Vec<Value>,
}

impl SqlFragment {
    pub fn new(sql: impl Into<String>, params: Vec<Value>) -> Self {
        Self {
            sql: sql.into(),
            params,
        }
    }
}

/// WHERE-fragment accumulator that keeps predicates and bind values in
/// the same order they are introduced.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SqlFilter {
    parts: Vec<String>,
    params: Vec<Value>,
}

impl SqlFilter {
    pub fn push_predicate(&mut self, sql: impl Into<String>) {
        self.parts.push(sql.into());
    }

    pub fn push_param(&mut self, value: Value) {
        self.params.push(value);
    }

    pub fn push_fragment(&mut self, fragment: SqlFragment) {
        if !fragment.sql.is_empty() {
            self.parts.push(fragment.sql);
        }
        self.params.extend(fragment.params);
    }

    pub fn where_clause(&self) -> String {
        where_clause(&self.parts)
    }

    pub fn into_params(self) -> Vec<Value> {
        self.params
    }
}

pub fn where_clause(predicates: &[String]) -> String {
    if predicates.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", predicates.join(" AND "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_keeps_predicate_and_param_order() {
        let mut filter = SqlFilter::default();
        filter.push_fragment(SqlFragment::new("start < ?", vec![Value::BigInt(20)]));
        filter.push_predicate("device_id = ?");
        filter.push_param(Value::Int(3));

        assert_eq!(filter.where_clause(), "WHERE start < ? AND device_id = ?");
        assert_eq!(
            filter.clone().into_params(),
            vec![Value::BigInt(20), Value::Int(3)]
        );
    }

    #[test]
    fn empty_filter_has_empty_where_clause() {
        let filter = SqlFilter::default();
        assert_eq!(filter.where_clause(), "");
        assert!(filter.into_params().is_empty());
    }

    #[test]
    fn where_clause_joins_predicates_or_empty() {
        assert_eq!(where_clause(&[]), "");
        assert_eq!(
            where_clause(&["a = ?".to_string(), "b > ?".to_string()]),
            "WHERE a = ? AND b > ?"
        );
    }
}
