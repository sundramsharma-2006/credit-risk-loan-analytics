"""build_pbip.py - generates the Power BI project (PBIP) from the CSVs in powerbi/data: semantic model (TMDL), relationships, DAX measures,
5-page report (all 8 PRD dashboard components) and theme.   Run:  python powerbi/build_pbip.py   (from the project root)
Then open powerbi/credora_dashboard.pbip in Power BI Desktop, set the DataPath parameter if the folder moved, and Refresh.
NOTE: generated from the PBIP/TMDL format used by the Zomato project (which opened in Power BI Desktop); this file has NOT been opened
in Power BI here - see README_BUILD.md for the manual build steps if anything fails to load."""
import json
import pathlib
import shutil
import uuid

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent; DATA = HERE / "data"; NAME = "credora_dashboard"
SM = HERE / f"{NAME}.SemanticModel"; RP = HERE / f"{NAME}.Report"
DATA_PATH = "C:\\Users\\hp\\OneDrive\\Desktop\\internship\\credit-risk-loan-analytics\\powerbi\\data"
DATE_COLS = {"dim_date": ["Date", "MonthStart"], "fact_loans": ["application_date", "decision_date", "application_month"]}
HIDE_SUM = True

MEASURES = {
    "fact_loans": [
        ("Applications", "COUNTROWS ( fact_loans )", "#,0", "KPIs"),
        ("Approved", "SUM ( fact_loans[IsApproved] )", "#,0", "KPIs"),
        ("Rejected", "SUM ( fact_loans[IsRejected] )", "#,0", "KPIs"),
        ("Approval Rate", "DIVIDE ( [Approved], [Approved] + [Rejected] )", "0.0%", "KPIs"),
        ("Disbursed Loans", "CALCULATE ( COUNTROWS ( fact_loans ), fact_loans[disbursed_flag] = 1 )", "#,0", "KPIs"),
        ("Defaults", "CALCULATE ( COUNTROWS ( fact_loans ), fact_loans[loan_default] = 1 )", "#,0", "KPIs"),
        ("Default Rate", "DIVIDE ( [Defaults], [Disbursed Loans] )", "0.0%", "KPIs"),
        ("Disbursed Value (Cr)", "SUM ( fact_loans[approved_amount_crore] )", "#,0.0", "KPIs"),
        ("Avg Requested Amount", "AVERAGE ( fact_loans[requested_amount] )", "#,0", "KPIs"),
        ("Avg Applicant Credit Score", "AVERAGEX ( fact_loans, RELATED ( dim_customer[credit_score] ) )", "0", "Credit"),
        ("Avg Model Default Probability", "AVERAGE ( fact_loans[model_default_probability] )", "0.0%", "Model"),
    ],
    "dim_customer": [
        ("Customers", "COUNTROWS ( dim_customer )", "#,0", "Risk"),
        ("High Risk Customers", 'CALCULATE ( COUNTROWS ( dim_customer ), dim_customer[risk_category] = "High Risk" )', "#,0", "Risk"),
        ("Avg Risk Score", "AVERAGE ( dim_customer[risk_score] )", "0.0", "Risk"),
        ("Avg Credit Score", "AVERAGE ( dim_customer[credit_score] )", "0", "Credit"),
    ],
    "ml_feature_importance": [("Driver Importance", "SUM ( ml_feature_importance[auc_drop_when_shuffled] )", "0.000", "Model")],
}
REL = [("rel_loans_date", "fact_loans.application_date", "dim_date.Date"), ("rel_loans_customer", "fact_loans.customer_id", "dim_customer.customer_id")]
TABLES = ["dim_date", "dim_customer", "fact_loans", "ml_model_comparison", "ml_feature_importance"]


def m_type(s, col, tb):
    if col in DATE_COLS.get(tb, []): return "type date", "dateTime", "Long Date"
    if pd.api.types.is_integer_dtype(s): return "Int64.Type", "int64", "0"
    if pd.api.types.is_float_dtype(s): return "type number", "double", None
    return "type text", "string", None


def table_tmdl(tb):
    df = pd.read_csv(DATA / f"{tb}.csv", nrows=5000); out = [f"table {tb}"]
    if tb == "dim_date": out.append("\tdataCategory: Time")
    out.append("")
    types = []
    for col in df.columns:
        mt, dt, fmt = m_type(pd.read_csv(DATA / f"{tb}.csv", usecols=[col])[col], col, tb); types.append((col, mt)); out += [f"\tcolumn {col}", f"\t\tdataType: {dt}"]
        if fmt: out.append(f"\t\tformatString: {fmt}")
        if tb == "dim_date" and col == "Date": out.append("\t\tisKey")
        out += ["\t\tsummarizeBy: none", f"\t\tsourceColumn: {col}", ""]
    for n, expr, fmt, folder in MEASURES.get(tb, []): out += [f"\tmeasure '{n}' = {expr}", f"\t\tformatString: {fmt}", f"\t\tdisplayFolder: {folder}", ""]
    tl = ", ".join('{"%s", %s}' % (c, t) for c, t in types)
    out += [f"\tpartition {tb} = m", "\t\tmode: import", "\t\tsource =", "\t\t\t\tlet",
            '\t\t\t\t\tSource = Csv.Document(File.Contents(DataPath & "\\' + tb + '.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
            "\t\t\t\t\tPromoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),", f'\t\t\t\t\tTyped = Table.TransformColumnTypes(Promoted, {{{tl}}}, "en-US")', "\t\t\t\tin", "\t\t\t\t\tTyped", ""]
    return "\n".join(out)


# ----------------------------------------------------------------------------- report helpers
_ctr = [0]
def _id():
    _ctr[0] += 1; return uuid.uuid5(uuid.NAMESPACE_URL, f"credora-visual-{_ctr[0]}").hex[:20]


def col(e, p): return {"Column": {"Expression": {"SourceRef": {"Entity": e}}, "Property": p}}
def mea(e, p): return {"Measure": {"Expression": {"SourceRef": {"Entity": e}}, "Property": p}}
def proj(f, active=False):
    k = "Column" if "Column" in f else "Measure"; e = f[k]["Expression"]["SourceRef"]["Entity"]; p = f[k]["Property"]
    d = {"field": f, "queryRef": f"{e}.{p}", "nativeQueryRef": p}
    if active: d["active"] = True
    return d
def lit(v): return {"expr": {"Literal": {"Value": v}}}
NOBORDER = {"border": [{"properties": {"show": lit("false")}}]}


def visual(vtype, x, y, w, h, z, query=None, title=None, sort=None, extra=None, objects=None):
    v = {"visualType": vtype}
    if query: v["query"] = {"queryState": query}
    if sort: v.setdefault("query", {})["sortDefinition"] = {"sort": [{"field": sort, "direction": "Descending"}]}
    if objects: v["objects"] = objects
    vc = dict(NOBORDER)
    if title: vc["title"] = [{"properties": {"show": lit("true"), "text": lit(f"'{title}'")}}]
    v["visualContainerObjects"] = vc; v["drillFilterOtherVisuals"] = True
    if extra: v.update(extra)
    return {"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.0.0/schema.json", "name": _id(),
            "position": {"x": x, "y": y, "z": z, "height": h, "width": w, "tabOrder": z}, "visual": v}


def card(m, x, y, w=195, h=88, z=4000, ent="fact_loans"): return visual("card", x, y, w, h, z, {"Values": {"projections": [proj(mea(ent, m))]}})
def slicer(e, p, x, y, w, z): return visual("slicer", x, y, w, 52, z, {"Values": {"projections": [proj(col(e, p))]}}, objects={"data": [{"properties": {"mode": lit("'Dropdown'")}}]}, extra={"syncGroup": {"groupName": p, "fieldChanges": True, "filterChanges": True}})
def chart(vt, cat, vals, x, y, w, h, z, title, sort=None): return visual(vt, x, y, w, h, z, {"Category": {"projections": [proj(cat, True)]}, "Y": {"projections": [proj(v) for v in vals]}}, title, sort)
def table(fields, x, y, w, h, z, title, sort=None): return visual("tableEx", x, y, w, h, z, {"Values": {"projections": [proj(f) for f in fields]}}, title, sort)
def text(t, x, y, w, h, z, size="22pt"):
    v = visual("textbox", x, y, w, h, z); v["visual"]["objects"] = {"general": [{"properties": {"paragraphs": [{"textRuns": [{"value": t, "textStyle": {"fontSize": size}}]}]}}]}
    v["visual"]["visualContainerObjects"] = {"title": [{"properties": {"show": lit("false")}}], "background": [{"properties": {"show": lit("false")}}], "border": [{"properties": {"show": lit("false")}}]}; v["position"].pop("tabOrder", None); return v


def pages():
    _ctr[0] = 0
    D, F, C = "dim_date", "fact_loans", "dim_customer"; month = col(D, "MonthStart"); P = []
    p1 = [text("1. Executive Overview", 20, 8, 820, 38, 1000), slicer(F, "loan_type", 640, 8, 190, 2000), slicer(C, "state", 840, 8, 200, 3000), slicer(C, "risk_category", 1050, 8, 210, 3100)]
    for i, m in enumerate(["Applications", "Approval Rate", "Default Rate", "Disbursed Value (Cr)"]): p1.append(card(m, 20 + i * 205, 70, z=4000 + i))
    p1 += [card("High Risk Customers", 840, 70, 205, 88, 4500, C), card("Avg Risk Score", 1055, 70, 205, 88, 4600, C),
           chart("lineChart", month, [mea(F, "Approval Rate")], 20, 175, 610, 250, 5000, "Approval rate by month"), chart("clusteredColumnChart", month, [mea(F, "Applications")], 650, 175, 610, 250, 5100, "Monthly loan applications"),
           chart("clusteredColumnChart", month, [mea(F, "Approved"), mea(F, "Rejected")], 20, 440, 610, 260, 5200, "Approval vs rejection trend"), chart("donutChart", col(C, "risk_category"), [mea(C, "Customers")], 650, 440, 610, 260, 5300, "Customer risk distribution")]
    P.append(("1. Executive Overview", p1))
    p2 = [text("2. Credit & Risk", 20, 8, 820, 38, 1000), slicer(F, "loan_type", 840, 8, 200, 2000), slicer(C, "risk_category", 1050, 8, 210, 3000),
          chart("clusteredBarChart", col(F, "loan_type"), [mea(F, "Default Rate")], 20, 70, 610, 300, 5000, "Default rate by loan type", mea(F, "Default Rate")),
          chart("clusteredColumnChart", col(C, "credit_band"), [mea(F, "Default Rate")], 650, 70, 610, 300, 5100, "Default rate by credit-score band"),
          chart("clusteredColumnChart", col(C, "credit_band"), [mea(C, "Customers")], 20, 385, 610, 315, 5200, "Customers by credit-score band"),
          chart("lineChart", month, [mea(F, "Avg Applicant Credit Score")], 650, 385, 610, 315, 5300, "Avg applicant credit score by application month (cohorts, not a time series)")]
    P.append(("2. Credit & Risk", p2))
    p3 = [text("3. Geography", 20, 8, 820, 38, 1000), slicer(F, "loan_type", 840, 8, 200, 2000), slicer(C, "risk_category", 1050, 8, 210, 3000),
          chart("clusteredBarChart", col(C, "state"), [mea(F, "Applications")], 20, 70, 610, 630, 5000, "Applications by state", mea(F, "Applications")),
          chart("clusteredBarChart", col(C, "state"), [mea(F, "Approval Rate")], 650, 70, 610, 205, 5100, "Approval rate by state"),
          chart("clusteredBarChart", col(C, "state"), [mea(F, "Default Rate")], 650, 285, 610, 205, 5200, "Default rate by state"),
          chart("clusteredBarChart", col(C, "state"), [mea(F, "Disbursed Value (Cr)")], 650, 500, 610, 200, 5300, "Disbursed value by state (Rs crore)")]
    P.append(("3. Geography", p3))
    p4 = [text("4. High-Risk Watchlist", 20, 8, 600, 38, 1000), slicer(C, "risk_category", 640, 8, 190, 2000), slicer(C, "state", 840, 8, 200, 3000), slicer(C, "employment_type", 1050, 8, 210, 3100),
          card("Customers", 20, 70, 300, 88, 4000, C), card("High Risk Customers", 330, 70, 300, 88, 4100, C), card("Avg Risk Score", 640, 70, 300, 88, 4200, C),
          table([col(C, c_) for c_ in ["customer_id", "city", "state", "employment_type", "credit_score", "risk_score", "risk_category", "top_risk_factors"]], 20, 175, 1240, 525, 5000, "Customers ranked by risk score (contact details deliberately excluded)", col(C, "risk_score"))]
    P.append(("4. High-Risk Watchlist", p4))
    p5 = [text("5. Model Performance", 20, 8, 820, 38, 1000), table([col("ml_model_comparison", c_) for c_ in ["Model", "CV_AUC", "ROC_AUC", "Precision", "Recall", "F1"]], 20, 70, 1240, 230, 5000, "Model comparison (held-out customers)"),
          chart("clusteredBarChart", col("ml_feature_importance", "feature"), [mea("ml_feature_importance", "Driver Importance")], 20, 315, 810, 385, 5100, "Top drivers of default (AUC drop when shuffled)"),
          text("Honest read: test ROC-AUC is about 0.73, below the PRD target of 0.80. Use the score as a second opinion, not an automatic decision.", 850, 330, 410, 150, 5200, "13pt")]
    P.append(("5. Model Performance", p5)); return P


def mea_col(e, p): return {"Aggregation": {"Expression": {"Column": {"Expression": {"SourceRef": {"Entity": e}}, "Property": p}}, "Function": 0}}


def build(zomato_template=None):
    # deterministic ids + overwrite in place (re-runnable; the folder may be sync-locked so nothing is deleted)
    for d in [SM / "definition" / "tables", SM / "definition" / "cultures", RP / "definition" / "pages", RP / "StaticResources" / "RegisteredResources"]: d.mkdir(parents=True, exist_ok=True)
    (SM / "definition.pbism").write_text(json.dumps({"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json", "version": "4.0", "settings": {}}, indent=2))
    (SM / "definition" / "database.tmdl").write_text("database\n\tcompatibilityLevel: 1600\n"); (SM / "definition" / "cultures" / "en-US.tmdl").write_text("cultureInfo en-US\n")
    (SM / "definition" / "expressions.tmdl").write_text(f'expression DataPath = "{DATA_PATH}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n')
    (SM / "definition" / "model.tmdl").write_text("model Model\n\tculture: en-US\n\tdefaultPowerBIDataSourceVersion: powerBI_V3\n\tdiscourageImplicitMeasures\n\n" + "\n".join(f"ref table {t}" for t in TABLES) + "\n\nref culture en-US\n")
    (SM / "definition" / "relationships.tmdl").write_text("\n".join(f"relationship {n}\n\tfromColumn: {a}\n\ttoColumn: {b}\n" for n, a, b in REL))
    for tb in TABLES: (SM / "definition" / "tables" / f"{tb}.tmdl").write_text(table_tmdl(tb))
    (HERE / f"{NAME}.pbip").write_text(json.dumps({"$schema": "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json", "version": "1.0", "artifacts": [{"report": {"path": f"{NAME}.Report"}}], "settings": {"enableAutoRecovery": True}}, indent=2))
    (RP / "definition.pbir").write_text(json.dumps({"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json", "version": "4.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}}}, indent=2))
    th = {"name": "Credora Risk", "dataColors": ["#0B5FA5", "#2D3A4A", "#8FB8DE", "#C23B3B", "#E0A526", "#2E9E5B", "#6B7A8F", "#D9E6F2"], "background": "#FFFFFF", "foreground": "#2D3A4A", "tableAccent": "#0B5FA5", "good": "#2E9E5B", "neutral": "#E0A526", "bad": "#C23B3B",
          "maximum": "#0B5FA5", "center": "#8FB8DE", "minimum": "#D9E6F2", "textClasses": {"callout": {"fontSize": 28, "fontFace": "Segoe UI Semibold", "color": "#0B5FA5"}, "title": {"fontSize": 12, "fontFace": "Segoe UI Semibold", "color": "#2D3A4A"}, "header": {"fontSize": 12, "fontFace": "Segoe UI Semibold", "color": "#2D3A4A"}, "label": {"fontSize": 10, "fontFace": "Segoe UI", "color": "#2D3A4A"}}}
    (RP / "StaticResources" / "RegisteredResources" / "CredoraTheme.json").write_text(json.dumps(th, indent=2)); (HERE / "credora_theme.json").write_text(json.dumps(th, indent=2))
    (RP / "definition" / "version.json").write_text(json.dumps({"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json", "version": "2.0.0"}, indent=2))
    (RP / "definition" / "report.json").write_text(json.dumps({"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/3.0.0/schema.json", "themeCollection": {"customTheme": {"name": "CredoraTheme.json", "reportVersionAtImport": {"visual": "2.1.0", "report": "2.1.0", "page": "2.0.0"}, "type": "RegisteredResources"}},
        "resourcePackages": [{"name": "RegisteredResources", "type": "RegisteredResources", "items": [{"name": "CredoraTheme.json", "path": "CredoraTheme.json", "type": "CustomTheme"}]}], "settings": {"useStylableVisualContainerHeader": True, "exportDataMode": "AllowSummarized", "defaultDrillFilterOtherVisuals": True, "allowChangeFilterTypes": True, "useEnhancedTooltips": True}}, indent=2))
    order = []
    for title, vis in pages():
        pid = uuid.uuid5(uuid.NAMESPACE_URL, f"credora-page-{title}").hex[:20]; order.append(pid); pd_ = RP / "definition" / "pages" / pid; (pd_ / "visuals").mkdir(parents=True, exist_ok=True)
        (pd_ / "page.json").write_text(json.dumps({"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.0.0/schema.json", "name": pid, "displayName": title, "displayOption": "FitToPage", "height": 720, "width": 1280}, indent=2))
        for v in vis: (pd_ / "visuals" / v["name"]).mkdir(exist_ok=True); (pd_ / "visuals" / v["name"] / "visual.json").write_text(json.dumps(v, indent=2))
    (RP / "definition" / "pages" / "pages.json").write_text(json.dumps({"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.0.0/schema.json", "pageOrder": order, "activePageName": order[0]}, indent=2))
    (HERE / ".gitignore").write_text("**/.pbi/localSettings.json\n**/.pbi/cache.abf\n"); return len(order)


if __name__ == "__main__":
    print("pages:", build())
