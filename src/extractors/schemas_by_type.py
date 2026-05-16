"""Required-fields catalog per document type."""

REQUIRED_FIELDS: dict[str, list[str]] = {
    "invoice": ["invoice_number", "vendor", "vendor_tax_id", "issue_date",
                "currency", "subtotal", "tax", "total", "payment_terms"],
    "contract": ["contract_number", "parties", "effective_date", "term_months",
                  "total_value", "currency"],
    "form":     ["form_id", "submitter", "date", "fields"],
    "report":   ["title", "author", "date", "summary"],
    "receipt":  ["receipt_number", "vendor", "issue_date", "total", "currency", "payment_method"],
}
