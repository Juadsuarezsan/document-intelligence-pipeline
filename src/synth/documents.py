"""Synthetic invoices, receipts, contracts, forms and reports with ground truth.

Each generator picks one of several layout templates (label spellings, date
formats, language, noise lines) so the heuristic extractor is not evaluated on
a single trivial pattern. Ground truth holds normalised values (ISO dates,
floats) for exactly the fields present in the text.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from src.schemas.document import DocumentType
from src.validators.regex_rules import nit_check_digit

SEED = 20260929
"""Default seed used by ``scripts/generate_synthetic.py``."""

VENDORS = [
    "Acme Consulting S.A.S.",
    "Blueharbor Analytics Ltd",
    "Forgecraft Ironworks",
    "Northwind Trading Co",
    "Tidewater Supplies Inc",
    "Vertex Agro",
    "Meridian Printworks",
    "Andes Logistics S.A.",
    "Solstice Medical Devices",
    "Quarry Lane Bakery",
    "Helix Software GmbH",
    "Cordillera Textiles",
]
CLIENTS = [
    "Globex Corporation",
    "Initech LLC",
    "Umbrella Health",
    "Stark Industries",
    "Wayne Enterprises",
    "Hooli Inc",
    "Pied Piper",
    "Vandelay Importers",
]
PEOPLE = [
    "Maria Fernanda Rojas",
    "John A. Whitaker",
    "Camila Restrepo",
    "Daniel Ortiz",
    "Priya Natarajan",
    "Luis Alberto Gómez",
    "Hannah Schmidt",
    "Tomás Herrera",
]
DEPARTMENTS = ["Finance", "Human Resources", "Operations", "Research", "Procurement", "Legal"]
CITIES = ["Bogotá", "Medellín", "Austin", "Berlin", "Lima", "Toronto"]
ITEMS = [
    ("Consulting hours", 120.0),
    ("Cloud hosting", 340.0),
    ("Steel beams", 89.5),
    ("Office chairs", 149.99),
    ("Printing services", 12.75),
    ("Coffee beans 1kg", 18.4),
    ("Software licence", 499.0),
    ("Freight", 210.0),
    ("Maintenance visit", 75.0),
    ("Fertiliser 25kg", 32.0),
]
CURRENCIES = ["USD", "EUR", "COP", "GBP", "MXN"]
LAWS = ["State of Delaware", "England and Wales", "Republic of Colombia", "Germany"]


@dataclass
class TableSpec:
    """A table rendered into the PDF version of a document."""

    headers: list[str]
    rows: list[list[str]]
    ruled: bool = True


@dataclass
class SynthDoc:
    """A generated document with its ground truth."""

    id: str
    doc_type: DocumentType
    template: str
    text: str
    ground_truth: dict[str, Any]
    table: TableSpec | None = None
    notes: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Serialise for ``test_set.jsonl``."""
        return {
            "id": self.id,
            "source": "synthetic",
            "doc_type": self.doc_type,
            "template": self.template,
            "text": self.text,
            "ground_truth": self.ground_truth,
            "table": (
                None
                if self.table is None
                else {
                    "headers": self.table.headers,
                    "rows": self.table.rows,
                    "ruled": self.table.ruled,
                }
            ),
            "notes": self.notes,
        }


def _money(v: float, style: str) -> str:
    if style == "eu":
        s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    return f"{v:,.2f}"


def _fmt_date(d: date, style: str) -> str:
    if style == "iso":
        return d.isoformat()
    if style == "dmy":
        return d.strftime("%d/%m/%Y")
    if style == "long":
        return d.strftime("%B %d, %Y")
    return d.strftime("%d %B %Y")


def valid_nit(rng: random.Random) -> str:
    """Generate a syntactically valid Colombian NIT with its check digit."""
    base = str(rng.randint(800_000_000, 901_999_999))
    return f"{base[:3]}.{base[3:6]}.{base[6:]}-{nit_check_digit(base)}"


def _line_items(rng: random.Random, n: int) -> tuple[list[list[str]], float]:
    rows: list[list[str]] = []
    subtotal = 0.0
    for name, unit in rng.sample(ITEMS, n):
        qty = rng.randint(1, 12)
        amount = round(qty * unit, 2)
        subtotal += amount
        rows.append([name, str(qty), f"{unit:.2f}", f"{amount:.2f}"])
    return rows, round(subtotal, 2)


def gen_invoice(rng: random.Random, idx: int) -> SynthDoc:
    """Invoice with one of four layouts (key/value, Spanish, block, prose)."""
    template = rng.choice(["kv_en", "es", "block", "prose"])
    vendor = rng.choice(VENDORS)
    client = rng.choice(CLIENTS)
    nit = valid_nit(rng)
    number = f"{rng.choice(['INV', 'FC', 'BH', 'TW'])}-{rng.randint(2024, 2026)}-{rng.randint(10, 9999):04d}"
    issue = date(2026, rng.randint(1, 9), rng.randint(1, 28))
    due = issue + timedelta(days=rng.choice([15, 30, 45, 60]))
    currency = rng.choice(CURRENCIES)
    rows, subtotal = _line_items(rng, rng.randint(2, 4))
    tax_rate = rng.choice([0.0, 0.05, 0.19, 0.21])
    tax = round(subtotal * tax_rate, 2)
    total = round(subtotal + tax, 2)
    terms = f"Net {(due - issue).days}"
    money_style = "eu" if template == "es" else "us"
    date_style = (
        rng.choice(["iso", "dmy", "long"]) if template != "es" else rng.choice(["iso", "dmy"])
    )
    gt: dict[str, Any] = {
        "invoice_number": number,
        "vendor": vendor,
        "vendor_tax_id": nit,
        "issue_date": issue.isoformat(),
        "due_date": due.isoformat(),
        "currency": currency,
        "subtotal": subtotal,
        "tax": tax,
        "total": total,
        "payment_terms": terms,
    }
    if template == "kv_en":
        lines = [
            vendor.upper(),
            f"{rng.choice(CITIES)} - {rng.choice(['Calle', 'Av.', 'Road'])} {rng.randint(1, 200)}",
            f"NIT: {nit}",
            "",
            f"Invoice No: {number}",
            f"Invoice Date: {_fmt_date(issue, date_style)}",
            f"Due Date: {_fmt_date(due, date_style)}",
            f"Bill To: {client}",
            f"Payment Terms: {terms}",
            "",
            "Description | Qty | Unit | Amount",
        ]
        lines += [" | ".join(r) for r in rows]
        lines += [
            "",
            f"Subtotal: {_money(subtotal, money_style)}",
            f"Tax ({int(tax_rate * 100)}%): {_money(tax, money_style)}",
            f"Total Due: {currency} {_money(total, money_style)}",
            f"Currency: {currency}",
        ]
    elif template == "es":
        lines = [
            "FACTURA DE VENTA",
            vendor,
            f"NIT: {nit}",
            f"Factura No: {number}",
            f"Fecha: {_fmt_date(issue, date_style)}",
            f"Vencimiento: {_fmt_date(due, date_style)}",
            f"Cliente: {client}",
            f"Moneda: {currency}",
            "",
        ]
        lines += [f"{r[0]}  x{r[1]}  {r[3]}" for r in rows]
        lines += [
            "",
            f"Subtotal: {_money(subtotal, money_style)}",
            f"IVA: {_money(tax, money_style)}",
            f"Total a pagar: {_money(total, money_style)}",
            f"Condiciones de pago: {terms}",
        ]
    elif template == "prose":
        lines = [
            "INVOICE",
            "",
            f"Invoice {number} was issued by {vendor} (tax id {nit}) on "
            f"{_fmt_date(issue, 'long')} to {client}.",
            f"The amount of {currency} {_money(total, money_style)} (subtotal "
            f"{_money(subtotal, money_style)} plus tax {_money(tax, money_style)}) is payable "
            f"by {_fmt_date(due, 'long')} under {terms} terms.",
            "",
            "Items: " + "; ".join(f"{r[0]} x{r[1]} = {r[3]}" for r in rows),
        ]
    else:
        lines = [
            "INVOICE",
            "",
            f"From: {vendor}",
            f"Tax ID: {nit}",
            f"To: {client}",
            "",
            f"Invoice #: {number}",
            f"Date: {_fmt_date(issue, date_style)}",
            f"Due: {_fmt_date(due, date_style)}",
            f"Terms: {terms}",
            "",
        ]
        lines += [f"{r[0]:<24}{r[1]:>4}{r[2]:>10}{r[3]:>12}" for r in rows]
        lines += [
            "",
            f"Sub Total: {_money(subtotal, money_style)}",
            f"VAT: {_money(tax, money_style)}",
            f"Amount Due: {_money(total, money_style)} {currency}",
            "",
            "Thank you for your business.",
        ]
    notes: list[str] = []
    if rng.random() < 0.2:
        # Deliberate arithmetic error so the validator has something to catch.
        gt["total"] = round(total + 10.0, 2)
        lines = [
            (
                ln.replace(_money(total, money_style), _money(gt["total"], money_style))
                if re.match(r"\s*(total|amount due|total a pagar)", ln, re.IGNORECASE)
                else ln
            )
            for ln in lines
        ]
        notes.append("math_mismatch_injected")
    table = TableSpec(
        headers=["Description", "Qty", "Unit price", "Amount"], rows=rows, ruled=rng.random() < 0.7
    )
    return SynthDoc(f"syn-inv-{idx:03d}", "invoice", template, "\n".join(lines), gt, table, notes)


def gen_receipt(rng: random.Random, idx: int) -> SynthDoc:
    """Point-of-sale receipt with three layouts (POS ticket, key/value, prose)."""
    template = rng.choice(["pos", "kv", "prose"])
    vendor = rng.choice(VENDORS)
    number = f"{rng.choice(['R', 'RC', 'T'])}-{rng.randint(100000, 999999)}"
    when = date(2026, rng.randint(1, 9), rng.randint(1, 28))
    currency = rng.choice(["USD", "EUR", "COP"])
    rows, subtotal = _line_items(rng, rng.randint(1, 3))
    total = round(subtotal, 2)
    method = rng.choice(["cash", "card", "visa", "mastercard", "transfer"])
    gt: dict[str, Any] = {
        "receipt_number": number,
        "vendor": vendor,
        "issue_date": when.isoformat(),
        "total": total,
        "currency": currency,
        "payment_method": method,
    }
    if template == "pos":
        lines = [
            vendor.upper(),
            f"{rng.choice(CITIES)}",
            "",
            f"Receipt #: {number}",
            f"Date: {_fmt_date(when, rng.choice(['iso', 'dmy']))}",
            "",
        ]
        lines += [f"{r[0]:<22} {r[1]:>2} x {r[2]:>8} {r[3]:>10}" for r in rows]
        lines += [
            "",
            f"TOTAL: {currency} {total:,.2f}",
            f"Paid by: {method.upper()}",
            "Change due: 0.00",
            "",
            "Thank you for shopping with us",
        ]
    elif template == "prose":
        lines = [
            f"Thank you for your purchase at {vendor} on {_fmt_date(when, 'long')}.",
            f"Transaction {number} for {total:,.2f} {currency} was paid by {method}.",
            "Keep this receipt for returns.",
        ]
    else:
        lines = [
            "PAYMENT RECEIPT",
            f"Merchant: {vendor}",
            f"Receipt Number: {number}",
            f"Date: {_fmt_date(when, rng.choice(['iso', 'long']))}",
            f"Amount: {total:,.2f} {currency}",
            f"Payment method: {method}",
            "Payment received - thank you.",
        ]
    return SynthDoc(f"syn-rcp-{idx:03d}", "receipt", template, "\n".join(lines), gt)


def gen_contract(rng: random.Random, idx: int) -> SynthDoc:
    """Services agreement with recital, schedule and letter layouts."""
    template = rng.choice(["recital", "schedule", "letter"])
    a, b = rng.choice(VENDORS), rng.choice(CLIENTS)
    number = (
        f"{rng.choice(['MSA', 'SOW', 'CTR'])}-{rng.randint(2024, 2026)}-{rng.randint(1, 999):03d}"
    )
    eff = date(2026, rng.randint(1, 9), rng.randint(1, 28))
    term = rng.choice([6, 12, 24, 36])
    value = float(rng.randint(10, 900) * 1000)
    currency = rng.choice(["USD", "EUR", "GBP"])
    law = rng.choice(LAWS)
    gt: dict[str, Any] = {
        "contract_number": number,
        "parties": f"{a}; {b}",
        "effective_date": eff.isoformat(),
        "term_months": term,
        "total_value": value,
        "currency": currency,
        "governing_law": law,
    }
    if template == "recital":
        lines = [
            "MASTER SERVICES AGREEMENT",
            f"Contract No: {number}",
            "",
            f"This Agreement is entered into as of {_fmt_date(eff, 'long')} (the Effective Date) between {a} and {b}.",
            "",
            f"1. Term. The initial term of this Agreement is {term} months from the Effective Date.",
            f"2. Fees. The total contract value is {currency} {value:,.2f}, payable in equal monthly instalments.",
            f"3. Governing Law: {law}.",
            "",
            "IN WITNESS WHEREOF the parties have executed this Agreement.",
        ]
    elif template == "letter":
        lines = [
            f"{rng.choice(CITIES)}, {_fmt_date(eff, 'long')}",
            "",
            f"Dear {rng.choice(PEOPLE)},",
            "",
            f"This letter confirms agreement {number} between {a} and {b}, commencing on "
            f"{_fmt_date(eff, 'dmy')} for a period of {term} months.",
            f"The total fee payable under this agreement is {currency} {value:,.2f}.",
            f"This agreement shall be governed by the laws of {law}.",
            "",
            "Yours faithfully,",
        ]
    else:
        lines = [
            "SERVICES CONTRACT - SCHEDULE A",
            f"Agreement Number: {number}",
            f"Parties: {a}; {b}",
            f"Effective Date: {_fmt_date(eff, rng.choice(['iso', 'dmy']))}",
            f"Term: {term} months",
            f"Total Value: {currency} {value:,.2f}",
            f"Currency: {currency}",
            f"Governing Law: {law}",
            "",
            "Signed by the authorised representatives of both parties.",
        ]
    return SynthDoc(f"syn-ctr-{idx:03d}", "contract", template, "\n".join(lines), gt)


def gen_form(rng: random.Random, idx: int) -> SynthDoc:
    """Administrative form with FUNSD-like labelled fields (one layout has no colons)."""
    template = rng.choice(["application", "request", "nocolon"])
    form_id = f"{rng.choice(['F', 'HR', 'PR'])}-{rng.randint(1000, 99999)}"
    person = rng.choice(PEOPLE)
    dept = rng.choice(DEPARTMENTS)
    when = date(2026, rng.randint(1, 9), rng.randint(1, 28))
    extra = {
        "Position": rng.choice(["Analyst", "Engineer", "Manager", "Technician"]),
        "Location": rng.choice(CITIES),
        "Phone": f"+57 {rng.randint(300, 350)} {rng.randint(100, 999)} {rng.randint(1000, 9999)}",
        "Email": f"{person.split()[0].lower()}@example.org",
    }
    gt: dict[str, Any] = {
        "form_id": form_id,
        "submitter": person,
        "date": when.isoformat(),
        "department": dept,
    }
    if template == "application":
        lines = [
            "APPLICATION FORM",
            "Please print clearly",
            "",
            f"Form ID: {form_id}",
            f"Name: {person}",
            f"Department: {dept}",
            f"Date: {_fmt_date(when, rng.choice(['iso', 'dmy']))}",
        ]
    elif template == "nocolon":
        lines = [
            "EMPLOYEE DATA SHEET",
            f"Form ID   {form_id}",
            f"Name   {person}",
            f"Department   {dept}",
            f"Date   {_fmt_date(when, 'dmy')}",
        ]
    else:
        lines = [
            "INTERNAL REQUEST",
            f"Form No: {form_id}",
            f"Submitted by: {person}",
            f"Dept: {dept}",
            f"Submission date: {_fmt_date(when, rng.choice(['iso', 'long']))}",
        ]
    for k, v in extra.items():
        lines.append(f"{k}   {v}" if template == "nocolon" else f"{k}: {v}")
    lines += ["", "Signature: ____________________"]
    gt["pairs"] = [[k, v] for k, v in extra.items()]
    return SynthDoc(f"syn-frm-{idx:03d}", "form", template, "\n".join(lines), gt)


def gen_report(rng: random.Random, idx: int) -> SynthDoc:
    """Narrative report: annual, technical or memo layout."""
    template = rng.choice(["annual", "technical", "memo"])
    author = rng.choice(PEOPLE)
    when = date(2026, rng.randint(1, 9), rng.randint(1, 28))
    fy = str(rng.choice([2024, 2025]))
    org = rng.choice(VENDORS)
    title = (
        f"Annual Report {fy}"
        if template == "annual"
        else f"{rng.choice(['Site Audit', 'Load Test', 'Quality Review'])} Report"
    )
    gt: dict[str, Any] = {
        "title": title,
        "author": author,
        "date": when.isoformat(),
        "fiscal_year": fy,
    }
    if template == "annual":
        lines = [
            title.upper(),
            org,
            f"Prepared by: {author}",
            f"Date: {_fmt_date(when, rng.choice(['iso', 'long']))}",
            f"Fiscal Year: {fy}",
            "",
            "Executive Summary",
            f"Revenue grew {rng.randint(3, 25)}% year over year while operating costs remained flat.",
            "Table of Contents",
            "1. Highlights",
            "2. Financial statements",
        ]
    elif template == "memo":
        lines = [
            "MEMORANDUM",
            f"TO: {rng.choice(CLIENTS)}",
            f"FROM: {author}",
            f"DATE: {_fmt_date(when, 'long')}",
            f"RE: {title}",
            "",
            f"Summary of activities for fiscal year {fy} follows.",
        ]
    else:
        lines = [
            title,
            f"Author: {author}",
            f"Report date: {_fmt_date(when, rng.choice(['iso', 'dmy']))}",
            f"Period: FY{fy}",
            "",
            "Findings",
            f"- {rng.randint(2, 9)} issues were identified during the review.",
            "- No critical defects remain open.",
        ]
        gt["fiscal_year"] = fy
    return SynthDoc(f"syn-rpt-{idx:03d}", "report", template, "\n".join(lines), gt)


GENERATORS = {
    "invoice": gen_invoice,
    "receipt": gen_receipt,
    "contract": gen_contract,
    "form": gen_form,
    "report": gen_report,
}


def generate_documents(n_per_type: int = 20, seed: int = SEED) -> list[SynthDoc]:
    """Generate ``n_per_type`` documents of each type deterministically."""
    rng = random.Random(seed)
    docs: list[SynthDoc] = []
    for gen in GENERATORS.values():
        for i in range(1, n_per_type + 1):
            docs.append(gen(rng, i))
    return docs


def generate_tables(n: int = 30, seed: int = SEED + 1) -> list[dict[str, Any]]:
    """Generate table specs (ruled and unruled) for the cell-level F1 evaluation."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        kind = rng.choice(["items", "financial", "schedule"])
        if kind == "items":
            headers = ["Description", "Qty", "Unit price", "Amount"]
            rows, _ = _line_items(rng, rng.randint(2, 6))
        elif kind == "financial":
            headers = ["Line", "FY2024", "FY2025", "Change"]
            rows = []
            for name in rng.sample(
                ["Revenue", "COGS", "Gross profit", "Opex", "EBITDA", "Net income"],
                rng.randint(3, 6),
            ):
                a, b = rng.randint(100, 9000), rng.randint(100, 9000)
                rows.append([name, f"{a:,}", f"{b:,}", f"{(b - a) / max(a, 1) * 100:+.1f}%"])
        else:
            headers = ["Milestone", "Owner", "Due date", "Status"]
            rows = [
                [
                    f"Phase {k}",
                    rng.choice(PEOPLE).split()[0],
                    date(2026, rng.randint(1, 12), rng.randint(1, 28)).isoformat(),
                    rng.choice(["Open", "Done", "Blocked"]),
                ]
                for k in range(1, rng.randint(3, 6))
            ]
        out.append(
            {
                "id": f"syn-tbl-{i:03d}",
                "kind": kind,
                "headers": headers,
                "rows": rows,
                "ruled": i % 3 != 0,
            }
        )
    return out
