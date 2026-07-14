pub const REDACTED_VALUE: &str = "[REDACTED]";

pub fn redact_secret(value: &str) -> String {
    if value.is_empty() {
        return String::new();
    }
    REDACTED_VALUE.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_secret_values() {
        assert_eq!(redact_secret("super-secret"), REDACTED_VALUE);
    }
}
