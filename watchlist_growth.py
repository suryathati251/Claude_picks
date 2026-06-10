"""
watchlist_growth.py — ADDITIVE high-growth candidates.

These are appended to the original WATCHLIST (nothing removed). They're here to
feed the GROWTH/asymmetric score in fundamentals.py — names with hypergrowth
*characteristics* (fast revenue growth, scalable margins). They are candidates to
research, NOT predictions and NOT endorsements. Most high-growth names do not
10x; some lose most of their value. Size anything from this bucket small.

target is intentionally None for all of these: the screener now ranks on reported
fundamentals, not analyst price targets, so we don't bake in stale target numbers.
Duplicates of tickers already in the original WATCHLIST are deliberately omitted.
"""

GROWTH_WATCHLIST = [
    # ---- High-growth software / security / data ----
    {"ticker": "NET", "region": "US", "name": "Cloudflare", "sector": "Hypergrowth", "target": None,
     "thesis": "Edge network + zero-trust + developer platform; broad usage-based growth runway.",
     "catalyst": "Workers/AI inference at the edge; large-customer adds", "risk": "Rich multiple; profitability still thin"},
    {"ticker": "DDOG", "region": "US", "name": "Datadog", "sector": "Hypergrowth", "target": None,
     "thesis": "Observability platform land-and-expand; many modules per customer.",
     "catalyst": "New-product attach; AI/LLM observability", "risk": "Consumption model; cloud-spend sensitivity"},
    {"ticker": "ZS", "region": "US", "name": "Zscaler", "sector": "Hypergrowth", "target": None,
     "thesis": "Cloud-native zero-trust security; secular shift from legacy appliances.",
     "catalyst": "Emerging-product ARR; large-deal velocity", "risk": "Competition; valuation"},
    {"ticker": "MDB", "region": "US", "name": "MongoDB", "sector": "Hypergrowth", "target": None,
     "thesis": "Developer-favorite document database; Atlas consumption scaling.",
     "catalyst": "Atlas growth; AI-app workloads", "risk": "Consumption volatility; SBC dilution"},
    {"ticker": "TTD", "region": "US", "name": "The Trade Desk", "sector": "Hypergrowth", "target": None,
     "thesis": "Independent demand-side ad platform; CTV/streaming ad-buying tailwind.",
     "catalyst": "Kokai ramp; CTV ad budgets shifting", "risk": "Ad-cycle sensitivity; execution stumbles"},
    {"ticker": "IOT", "region": "US", "name": "Samsara", "sector": "Hypergrowth", "target": None,
     "thesis": "Connected-operations / IoT for physical industries; large untapped base.",
     "catalyst": "Large-customer ARR; new sensor lines", "risk": "Hardware-attached; macro-cyclical end markets"},
    {"ticker": "APP", "region": "US", "name": "AppLovin", "sector": "Hypergrowth", "target": None,
     "thesis": "AI-driven ad engine (AXON) monetizing mobile + expanding beyond gaming.",
     "catalyst": "Ad-engine expansion to e-commerce/CTV", "risk": "Concentration in ad-tech; volatility"},

    # ---- Fintech growth ----
    {"ticker": "NU", "region": "US", "name": "Nu Holdings", "sector": "Hypergrowth", "target": None,
     "thesis": "LatAm digital bank scaling members + product depth; profitable growth.",
     "catalyst": "Mexico/Colombia expansion; ARPU growth", "risk": "EM/currency risk; credit cycle"},
    {"ticker": "SOFI", "region": "US", "name": "SoFi Technologies", "sector": "Hypergrowth", "target": None,
     "thesis": "Digital one-stop bank; member + fee-based revenue growth; Rule-of-40 history.",
     "catalyst": "Fee-income mix shift; member growth", "risk": "High beta (~2.2); credit + rate sensitivity"},
    {"ticker": "HOOD", "region": "US", "name": "Robinhood", "sector": "Hypergrowth", "target": None,
     "thesis": "Broadening from brokerage to full financial platform; new revenue lines.",
     "catalyst": "Crypto/derivatives; new products; ARPU", "risk": "Trading-activity dependent; regulatory"},
    {"ticker": "MELI", "region": "US", "name": "MercadoLibre", "sector": "Hypergrowth", "target": None,
     "thesis": "LatAm e-commerce + fintech (Mercado Pago) compounding; long runway.",
     "catalyst": "Fintech TPV; logistics scale", "risk": "EM/currency; competition from Amazon/Shopee"},

    # ---- Other high-growth profiles the user has touched on ----
    {"ticker": "ESTC", "region": "US", "name": "Elastic", "sector": "Hypergrowth", "target": None,
     "thesis": "Search/observability platform; GenAI vector-search relevance.",
     "catalyst": "Elastic Cloud growth; AI search workloads", "risk": "Competitive search market"},
    {"ticker": "GRAB", "region": "US", "name": "Grab Holdings", "sector": "Hypergrowth", "target": None,
     "thesis": "SE-Asia super-app (deliveries + mobility + fintech) approaching scale profitability.",
     "catalyst": "Adjusted-EBITDA inflection; fintech lending", "risk": "EM/competitive; path-to-GAAP-profit"},
    {"ticker": "VRT", "region": "US", "name": "Vertiv Holdings", "sector": "Hypergrowth", "target": None,
     "thesis": "Data-center power & cooling — direct pick-and-shovel on AI capex buildout.",
     "catalyst": "AI data-center orders/backlog; liquid cooling", "risk": "Cyclical capex; AI-spend dependence"},
    {"ticker": "RKLB", "region": "US", "name": "Rocket Lab", "sector": "Hypergrowth", "target": None,
     "thesis": "Small-launch + space-systems; Neutron rocket optionality. Speculative high-beta.",
     "catalyst": "Neutron first launch; space-systems backlog", "risk": "Pre-scale; cash burn; binary launch risk"},
    {"ticker": "TOST", "region": "US", "name": "Toast", "sector": "Hypergrowth", "target": None,
     "thesis": "Restaurant-software OS; location growth + payments attach; crossed into GAAP profitability.",
     "catalyst": "Enterprise/international wins; fintech ARPU", "risk": "Restaurant churn in a downturn; thin margins"},
    {"ticker": "DUOL", "region": "US", "name": "Duolingo", "sector": "Hypergrowth", "target": None,
     "thesis": "Category-defining consumer learning app; 40%+ growth with real profits and net cash.",
     "catalyst": "Max (AI) subscription tier; new subjects", "risk": "Rich multiple; LLMs commoditizing language learning"},

    # ---- AI infrastructure / neoclouds (fastest top-line growth in the market) ----
    {"ticker": "CRWV", "region": "US", "name": "CoreWeave", "sector": "Hypergrowth", "target": None,
     "thesis": "AI neocloud; ~140%+ revenue growth, multi-year hyperscaler/AI-lab contracts; street sees ~$23B sales by 2027.",
     "catalyst": "New data-center capacity; contract backlog growth", "risk": "Heavy debt-funded capex; customer concentration; GPU depreciation"},
    {"ticker": "NBIS", "region": "US", "name": "Nebius Group", "sector": "Hypergrowth", "target": None,
     "thesis": "AI neocloud scaling from ~$1.25B to a guided $7–9B run-rate in 2026; triple-digit growth.",
     "catalyst": "Data-center buildout; large AI customer wins", "risk": "Execution on extreme scale-up; capex intensity; competition"},
    {"ticker": "ALAB", "region": "US", "name": "Astera Labs", "sector": "Hypergrowth", "target": None,
     "thesis": "AI connectivity silicon (PCIe/CXL retimers, Scorpio switches); ~90% revenue growth, S&P 500 candidate.",
     "catalyst": "Scorpio ramp with next-gen accelerators", "risk": "Rich multiple; hyperscaler design-win concentration"},
    {"ticker": "CRDO", "region": "US", "name": "Credo Technology", "sector": "Hypergrowth", "target": None,
     "thesis": "AI data-center connectivity (AECs/SerDes); ~150% revenue growth riding back-end network buildout.",
     "catalyst": "New hyperscaler qualifications; 800G/1.6T cycles", "risk": "Customer concentration; AI-capex sensitivity"},

    # ---- Consumer internet / platforms ----
    {"ticker": "RDDT", "region": "US", "name": "Reddit", "sector": "Hypergrowth", "target": None,
     "thesis": "~70% ad-revenue growth; unique content moat increasingly licensed for AI training; S&P 500 candidate.",
     "catalyst": "Ad-platform improvements; AI data-licensing deals", "risk": "Google-search traffic dependence; engagement volatility"},
    {"ticker": "HIMS", "region": "US", "name": "Hims & Hers Health", "sector": "Hypergrowth", "target": None,
     "thesis": "DTC telehealth compounding subscribers across categories (weight loss, dermatology, mental health).",
     "catalyst": "GLP-1 offering scale; new specialty launches", "risk": "Regulatory (compounded GLP-1s); marketing-spend dependence"},
    {"ticker": "CAVA", "region": "US", "name": "CAVA Group", "sector": "Hypergrowth", "target": None,
     "thesis": "Fast-casual Mediterranean chain in early national rollout; strong unit economics, Chipotle-playbook story.",
     "catalyst": "New-store openings; same-store sales comps", "risk": "Restaurant multiple compression; consumer slowdown"},

    # ---- Data / security / health-AI ----
    {"ticker": "RBRK", "region": "US", "name": "Rubrik", "sector": "Hypergrowth", "target": None,
     "thesis": "Cyber-resilience / data-security platform; high-40s% ARR growth as ransomware recovery becomes board-level spend.",
     "catalyst": "Large-enterprise wins; security-suite expansion", "risk": "Competitive backup/security market; SBC dilution"},
    {"ticker": "TEM", "region": "US", "name": "Tempus AI", "sector": "Hypergrowth", "target": None,
     "thesis": "AI precision-medicine platform (genomics + clinical data); insurance coverage expanding.",
     "catalyst": "Genomics volume growth; data-licensing deals", "risk": "Path to profitability; reimbursement risk"},
    {"ticker": "AXON", "region": "US", "name": "Axon Enterprise", "sector": "Hypergrowth", "target": None,
     "thesis": "Public-safety platform (Taser, body cams, evidence cloud, drones); ~30% compounder with deep moat.",
     "catalyst": "AI evidence tools; federal/international expansion", "risk": "Premium valuation; municipal budget cycles"},
]
