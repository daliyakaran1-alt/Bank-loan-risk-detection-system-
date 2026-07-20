"""
Report Generator Module.

Generates PDF and HTML reports for Business Intelligence and EDA.
"""

import logging
import os
from typing import Dict, Any, List
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Handles exporting and generating reports."""

    def __init__(self, output_dir: str = "reports/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "eda"), exist_ok=True)

    def generate_dataset_summary_md(self, overview: Dict[str, Any], filename: str = "dataset_summary.md"):
        """Generate a markdown file with the dataset summary."""
        logger.info(f"Generating markdown summary at {filename}")
        path = os.path.join(self.output_dir, filename)
        
        lines = [
            "# Dataset Overview Summary",
            "",
            f"**Total Records (Rows):** {overview.get('shape', (0,0))[0]:,}",
            f"**Total Features (Columns):** {overview.get('shape', (0,0))[1]:,}",
            f"**Memory Usage:** {overview.get('memory_usage', 0):.2f} MB",
            "",
            "## Target Distribution"
        ]
        
        for k, v in overview.get('target_distribution', {}).items():
            lines.append(f"- **{k}**: {v*100:.2f}%")
            
        lines.append("")
        lines.append("## Feature Types")
        lines.append(f"- **Numerical:** {len(overview.get('numerical_features', []))} features")
        lines.append(f"- **Categorical:** {len(overview.get('categorical_features', []))} features")
        lines.append(f"- **Boolean:** {len(overview.get('boolean_features', []))} features")
        
        lines.append("")
        lines.append("## High Cardinality Features (Top 10)")
        for feat in overview.get("high_cardinality", [])[:10]:
            lines.append(f"- {feat}")
            
        lines.append("")
        lines.append("## Missing Values Summary")
        for feat, cnt in list(overview.get("missing_values", {}).items())[:15]:
            lines.append(f"- **{feat}**: {cnt:,} missing")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
    def generate_pdf_report(self, title: str, content_lines: List[str], filename: str):
        """Generate a basic PDF report using reportlab."""
        path = os.path.join(self.output_dir, filename)
        doc = SimpleDocTemplate(path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Define a custom style if needed
        normal_style = styles["Normal"]
        title_style = styles["Title"]
        
        flowables = []
        
        # Add title
        flowables.append(Paragraph(title, title_style))
        flowables.append(Spacer(1, 12))
        
        # Add lines
        for line in content_lines:
            # Replace markdown-like bold syntax to HTML for reportlab, or just simple text
            # ReportLab Paragraph supports simple <b> tags
            line_clean = line.replace('**', '<b>', 1).replace('**', '</b>', 1) if line.count('**') >= 2 else line
            flowables.append(Paragraph(line_clean, normal_style))
            flowables.append(Spacer(1, 6))
            
        try:
            doc.build(flowables)
            logger.info(f"Generated PDF at {path}")
        except Exception as e:
            logger.error(f"Failed to generate PDF with ReportLab: {e}")
            with open(path.replace('.pdf', '.txt'), "w", encoding="utf-8") as f:
                f.write(f"{title}\n\n" + "\n".join(content_lines))

    def generate_executive_report(self, kpis: Dict[str, Any]):
        """Generate the Executive KPIs PDF Report."""
        lines = []
        for k, kpi in kpis.items():
            lines.append(f"<b>{kpi.name}:</b> {kpi.formatted_value}")
        self.generate_pdf_report("FinSight Executive Report", lines, "Executive_Report.pdf")

    def generate_business_insights_report(self, insights: List[Any]):
        """Generate the Business Insights PDF Report."""
        lines = []
        for ind, insight in enumerate(insights):
            lines.append(f"<b>Insight {ind+1} [{insight.insight_type.upper()}]:</b>")
            # Insights contain newlines
            for part in insight.content.split('\n'):
                if part.strip():
                    lines.append(part.strip())
            lines.append("")
        self.generate_pdf_report("FinSight Business Insights", lines, "Business_Insights.pdf")

    def generate_eda_html(self, sections: Dict[str, Any]):
        """Generate a comprehensive HTML EDA report."""
        path = os.path.join(self.output_dir, "EDA_Report.html")
        html_content = [
            "<html><head><title>EDA Report</title><style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "h1, h2 { color: #1E88E5; }",
            "</style></head><body>",
            "<h1>Enterprise Banking Exploratory Data Analysis</h1>"
        ]
        
        for name, section in sections.items():
            html_content.append(f"<h2>{name}</h2>")
            html_content.append(f"<p>Contains {len(section.charts)} visualizations (available in interactive dashboard).</p>")
            
        html_content.append("</body></html>")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))
        logger.info(f"Generated HTML report at {path}")

    def generate_loan_assessment_pdf(self, pred_id: str, score: int, prob: float, decision: str, explanation: str, recs: Dict[str, str], shap_drivers: List[Dict[str, Any]], filename: str = "Loan_Assessment_Report.pdf"):
        """Generates a professional Loan Assessment PDF for an individual applicant."""
        path = os.path.join(self.output_dir, filename)
        
        # Prepare content lines
        content = [
            "**FinSight Enterprise Banking Platform**",
            f"**Report ID:** {pred_id}",
            f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "### 1. Risk Assessment",
            f"**Assigned Risk Score:** {score} (300-900 Scale)",
            f"**Default Probability:** {prob:.2%}",
            f"**System Decision:** {decision}",
            "",
            "### 2. Underwriting Rationale",
            explanation,
            "",
            "### 3. Key Risk Drivers (SHAP Analysis)",
        ]
        
        for idx, driver in enumerate(shap_drivers[:3]):
            content.append(f"- **{driver['Feature']}**: {driver['Value']} (Impact: {driver['SHAP_Value']:.4f})")
            
        content.extend([
            "",
            "### 4. Strategic Recommendations",
            f"- **Execution Strategy:** {recs.get('Strategy', 'N/A')}",
            f"- **Pricing Tier:** {recs.get('Interest Rate', 'N/A')}",
            f"- **Monitoring Frequency:** {recs.get('Monitoring', 'N/A')}",
            "",
            "---",
            "*This report is automatically generated by the FinSight ML Engine. Final underwriting decisions remain subject to bank policy.*"
        ])
        
        self.generate_pdf_report(f"Loan Assessment Report", content, filename)

