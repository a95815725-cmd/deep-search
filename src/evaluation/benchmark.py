"""
Benchmark suite for the Financial Deep Search agent.

Defines 15 hard benchmark questions spanning five categories:
  - multi_hop        (require connecting facts from 2+ document sections)
  - temporal         (require year-over-year or trend reasoning)
  - numerical        (require exact extraction or calculation)
  - cross_section    (compare different document types or companies)
  - risk_analysis    (synthesise qualitative risk with quantitative data)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class BenchmarkQuestion:
    id: str
    question: str
    category: str  # multi_hop | temporal | numerical | cross_section | risk_analysis
    difficulty: str  # medium | hard | very_hard
    expected_sections: List[str]  # document sections expected to be retrieved
    ground_truth_hints: List[str]  # key facts / numbers that a good answer must contain
    requires_calculation: bool
    requires_multiple_years: bool
    notes: str = ""  # optional analyst commentary on the question


# --------------------------------------------------------------------------- #
# The 15 benchmark questions
# --------------------------------------------------------------------------- #

BENCHMARK_QUESTIONS: List[BenchmarkQuestion] = [
    # ===== MULTI-HOP (4 questions) =====
    BenchmarkQuestion(
        id="mh_001",
        question=(
            "How has the company's CET1 (Common Equity Tier 1) capital ratio trended "
            "over the past three reporting years, and which specific risk factors "
            "identified in the annual report could plausibly threaten this trend going forward?"
        ),
        category="multi_hop",
        difficulty="hard",
        expected_sections=["balance_sheet", "capital_adequacy", "risk_factors"],
        ground_truth_hints=[
            "CET1 ratio",
            "risk-weighted assets",
            "regulatory capital",
            "Basel III",
            "minimum capital requirement",
        ],
        requires_calculation=True,
        requires_multiple_years=True,
        notes="Requires linking capital table to qualitative risk narrative.",
    ),
    BenchmarkQuestion(
        id="mh_002",
        question=(
            "The management commentary mentions a strategic shift toward fee-based income. "
            "Does the income statement data for the last two years support this claim? "
            "Calculate the change in net fee & commission income as a percentage of total operating income."
        ),
        category="multi_hop",
        difficulty="hard",
        expected_sections=["income_statement", "management_discussion"],
        ground_truth_hints=[
            "net fee and commission income",
            "total operating income",
            "percentage of income",
            "non-interest income",
        ],
        requires_calculation=True,
        requires_multiple_years=True,
        notes="Tests whether model can verify management claims against reported numbers.",
    ),
    BenchmarkQuestion(
        id="mh_003",
        question=(
            "What is the relationship between the reported loan loss provisions, "
            "the non-performing loan (NPL) ratio, and the level of loan-loss reserves on the balance sheet? "
            "Are the provisions consistent with the change in NPL ratio year-over-year?"
        ),
        category="multi_hop",
        difficulty="very_hard",
        expected_sections=["income_statement", "balance_sheet", "credit_risk", "notes_to_accounts"],
        ground_truth_hints=[
            "loan loss provision",
            "non-performing loans",
            "coverage ratio",
            "impairment",
            "stage 3 loans",
        ],
        requires_calculation=True,
        requires_multiple_years=True,
        notes="Deep accounting cross-check across income statement, balance sheet and credit risk notes.",
    ),
    BenchmarkQuestion(
        id="mh_004",
        question=(
            "How does the bank's interest rate risk exposure described in the market risk section "
            "compare to the actual year-over-year change in net interest margin (NIM)? "
            "Did the reported NIM move in the direction the risk disclosures suggested it might?"
        ),
        category="multi_hop",
        difficulty="very_hard",
        expected_sections=["market_risk", "income_statement", "interest_rate_risk"],
        ground_truth_hints=[
            "net interest margin",
            "interest rate sensitivity",
            "NIM",
            "repricing gap",
            "basis risk",
        ],
        requires_calculation=True,
        requires_multiple_years=True,
        notes="Requires matching forward-looking risk language to backward-looking financial results.",
    ),
    # ===== TEMPORAL (3 questions) =====
    BenchmarkQuestion(
        id="tm_001",
        question=(
            "Trace the evolution of the bank's return on equity (ROE) over the last "
            "three fiscal years. What were the primary drivers cited by management for "
            "the single largest year-over-year change?"
        ),
        category="temporal",
        difficulty="hard",
        expected_sections=["income_statement", "equity", "management_discussion"],
        ground_truth_hints=[
            "return on equity",
            "ROE",
            "net profit",
            "shareholders equity",
            "cost of equity",
        ],
        requires_calculation=True,
        requires_multiple_years=True,
        notes="Needs multi-year income statement and management narrative.",
    ),
    BenchmarkQuestion(
        id="tm_002",
        question=(
            "How have total risk-weighted assets (RWA) changed across all available "
            "annual reports, broken down by credit risk, market risk, and operational risk RWA? "
            "Which component grew fastest in percentage terms?"
        ),
        category="temporal",
        difficulty="hard",
        expected_sections=["capital_adequacy", "risk_weighted_assets"],
        ground_truth_hints=[
            "risk-weighted assets",
            "credit RWA",
            "market risk RWA",
            "operational risk RWA",
            "Pillar 1",
        ],
        requires_calculation=True,
        requires_multiple_years=True,
        notes="Requires RWA decomposition across years; may need Pillar 3 disclosures.",
    ),
    BenchmarkQuestion(
        id="tm_003",
        question=(
            "Compare the bank's cost-to-income ratio trend against its stated efficiency "
            "improvement targets. Is the bank on track to hit its target, and by how many "
            "basis points has the ratio improved or deteriorated each year?"
        ),
        category="temporal",
        difficulty="hard",
        expected_sections=["income_statement", "management_discussion", "strategy"],
        ground_truth_hints=[
            "cost-to-income ratio",
            "efficiency ratio",
            "operating expenses",
            "operating income",
            "efficiency target",
        ],
        requires_calculation=True,
        requires_multiple_years=True,
        notes="Links reported numbers to forward guidance / stated targets.",
    ),
    # ===== NUMERICAL (3 questions) =====
    BenchmarkQuestion(
        id="nu_001",
        question=(
            "Calculate the bank's loan-to-deposit ratio for the most recent fiscal year. "
            "Show your working using the balance sheet figures, and state whether this "
            "is above or below the commonly cited 80–100% prudential guideline."
        ),
        category="numerical",
        difficulty="medium",
        expected_sections=["balance_sheet"],
        ground_truth_hints=[
            "total loans",
            "total deposits",
            "loan-to-deposit ratio",
            "customer deposits",
            "net loans",
        ],
        requires_calculation=True,
        requires_multiple_years=False,
        notes="Straightforward ratio calculation; tests precision of number extraction.",
    ),
    BenchmarkQuestion(
        id="nu_002",
        question=(
            "What is the exact net interest income (NII) figure for each available year? "
            "Calculate the compound annual growth rate (CAGR) of NII over the full period "
            "covered by the available documents."
        ),
        category="numerical",
        difficulty="hard",
        expected_sections=["income_statement"],
        ground_truth_hints=[
            "net interest income",
            "NII",
            "interest income",
            "interest expense",
            "CAGR",
        ],
        requires_calculation=True,
        requires_multiple_years=True,
        notes="Tests accurate extraction across years and CAGR formula application.",
    ),
    BenchmarkQuestion(
        id="nu_003",
        question=(
            "What is the bank's total exposure to sovereign debt, and what percentage "
            "of its total asset base does this represent? Break down by geography if the "
            "data is available."
        ),
        category="numerical",
        difficulty="very_hard",
        expected_sections=["balance_sheet", "credit_risk", "country_risk", "notes_to_accounts"],
        ground_truth_hints=[
            "sovereign exposure",
            "government bonds",
            "total assets",
            "geographic breakdown",
            "country concentration",
        ],
        requires_calculation=True,
        requires_multiple_years=False,
        notes="Sovereign exposure data is often buried in credit risk or country risk notes.",
    ),
    # ===== CROSS-SECTION (3 questions) =====
    BenchmarkQuestion(
        id="cs_001",
        question=(
            "Compare the credit quality of the retail lending portfolio versus the "
            "wholesale / corporate lending portfolio. Which has a higher NPL ratio, "
            "and which has a higher provision coverage ratio?"
        ),
        category="cross_section",
        difficulty="hard",
        expected_sections=["credit_risk", "loan_portfolio", "segment_reporting"],
        ground_truth_hints=[
            "retail portfolio",
            "corporate portfolio",
            "NPL ratio",
            "coverage ratio",
            "stage 2",
            "stage 3",
        ],
        requires_calculation=True,
        requires_multiple_years=False,
        notes="Requires segment-level credit quality data, often in credit risk notes.",
    ),
    BenchmarkQuestion(
        id="cs_002",
        question=(
            "How does the bank's Pillar 2 additional capital requirement compare to "
            "its Pillar 1 minimum requirement, and what specific risks does the regulator's "
            "Supervisory Review and Evaluation Process (SREP) addendum address?"
        ),
        category="cross_section",
        difficulty="very_hard",
        expected_sections=["capital_adequacy", "regulatory_capital", "risk_factors"],
        ground_truth_hints=[
            "Pillar 2 requirement",
            "SREP",
            "Pillar 1",
            "combined buffer requirement",
            "P2R",
            "P2G",
            "total SREP capital requirement",
        ],
        requires_calculation=False,
        requires_multiple_years=False,
        notes="Requires understanding of Basel III / CRR regulatory framework.",
    ),
    BenchmarkQuestion(
        id="cs_003",
        question=(
            "How does the bank's disclosed climate-related financial risk (physical and "
            "transition risk) in the sustainability / ESG section relate to the credit "
            "risk concentrations disclosed in the credit risk section? "
            "Are the highest-risk climate sectors also the largest credit concentrations?"
        ),
        category="cross_section",
        difficulty="very_hard",
        expected_sections=["esg_sustainability", "credit_risk", "climate_risk"],
        ground_truth_hints=[
            "physical risk",
            "transition risk",
            "carbon-intensive sectors",
            "sector concentration",
            "TCFD",
            "financed emissions",
        ],
        requires_calculation=False,
        requires_multiple_years=False,
        notes="Emerging disclosure area; tests ability to link ESG and credit risk sections.",
    ),
    # ===== RISK ANALYSIS (2 questions) =====
    BenchmarkQuestion(
        id="ra_001",
        question=(
            "Based on all risk factor disclosures, identify the top three risks the bank "
            "considers most material and explain how each is quantified or mitigated. "
            "Then cross-reference whether the quantified risk metrics (e.g. VaR, stress loss) "
            "appear in the financial statements."
        ),
        category="risk_analysis",
        difficulty="very_hard",
        expected_sections=["risk_factors", "market_risk", "operational_risk", "balance_sheet"],
        ground_truth_hints=[
            "value at risk",
            "stress testing",
            "risk appetite",
            "risk mitigation",
            "principal risks",
            "material risk",
        ],
        requires_calculation=False,
        requires_multiple_years=False,
        notes="Holistic risk question requiring synthesis of qualitative and quantitative disclosures.",
    ),
    BenchmarkQuestion(
        id="ra_002",
        question=(
            "The bank's liquidity coverage ratio (LCR) and net stable funding ratio (NSFR) "
            "are reported in the liquidity risk section. How have these evolved over the "
            "reporting period, and at what level do they stand relative to regulatory minimums? "
            "What contingency funding sources are disclosed?"
        ),
        category="risk_analysis",
        difficulty="hard",
        expected_sections=["liquidity_risk", "capital_adequacy", "risk_factors"],
        ground_truth_hints=[
            "liquidity coverage ratio",
            "LCR",
            "net stable funding ratio",
            "NSFR",
            "100% minimum",
            "high-quality liquid assets",
            "HQLA",
            "contingency funding",
        ],
        requires_calculation=False,
        requires_multiple_years=True,
        notes="Liquidity regulatory ratios are often in a dedicated liquidity risk note.",
    ),
]

# Sanity check at module load
assert len(BENCHMARK_QUESTIONS) == 15, (
    f"Expected 15 benchmark questions, got {len(BENCHMARK_QUESTIONS)}"
)

# Quick-access lookup by ID
QUESTIONS_BY_ID: Dict[str, BenchmarkQuestion] = {q.id: q for q in BENCHMARK_QUESTIONS}
