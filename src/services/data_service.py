"""
DataService: Provides real-time KPIs and analytics data from the Applicant database table.
All data comes from actual processed applications — no dummy/static data.
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from src.db.models import Applicant
from datetime import datetime, timedelta


class DataService:
    """Fetches real-time analytics data from the applicant database."""

    def __init__(self, db: Session, user_id: int):
        self._db = db
        self._user_id = user_id

    def get_dashboard_kpis(self) -> dict:
        """Calculate real KPIs from the applicant database."""
        total = self._db.query(func.count(Applicant.id)).filter(Applicant.user_id == self._user_id).scalar() or 0
        if total == 0:
            return {
                "total_customers": "0", "approved": "0", "rejected": "0",
                "default_rate": "0%", "high_risk": "0", "total_loans": "₹0"
            }

        approved = self._db.query(func.count(Applicant.id)).filter(Applicant.user_id == self._user_id, Applicant.recommendation == 'Approve Loan').scalar() or 0
        rejected = self._db.query(func.count(Applicant.id)).filter(Applicant.user_id == self._user_id, Applicant.recommendation == 'Reject Loan').scalar() or 0
        manual_review = total - approved - rejected
        high_risk = self._db.query(func.count(Applicant.id)).filter(Applicant.user_id == self._user_id, Applicant.risk_level == 'HIGH').scalar() or 0
        total_loan_amt = self._db.query(func.sum(Applicant.loan_amount)).filter(Applicant.user_id == self._user_id).scalar() or 0
        default_rate = (rejected / total) * 100 if total > 0 else 0

        return {
            "total_customers": f"{total:,}",
            "approved": f"{approved:,}",
            "rejected": f"{rejected:,}",
            "manual_review": f"{manual_review:,}",
            "default_rate": f"{default_rate:.1f}%",
            "high_risk": f"{high_risk:,}",
            "total_loans": f"₹{total_loan_amt / 10_000_000:.2f} Cr"
        }

    def get_analytics_data(self) -> dict:
        """Get charts data from the applicant database."""
        applicants = self._db.query(Applicant).filter(Applicant.user_id == self._user_id).all()
        if not applicants:
            return {
                "income_dist": {"labels": [], "data": []},
                "credit_default": {"labels": [], "data": []},
                "age_default": {"labels": [], "data": []}
            }

        # 1. Income Distribution
        brackets = {'< ₹1L': 0, '₹1L-3L': 0, '₹3L-5L': 0, '> ₹5L': 0}
        for a in applicants:
            inc = a.income or 0
            if inc < 100000:
                brackets['< ₹1L'] += 1
            elif inc < 300000:
                brackets['₹1L-3L'] += 1
            elif inc < 500000:
                brackets['₹3L-5L'] += 1
            else:
                brackets['> ₹5L'] += 1

        # 2. Credit Score vs Risk Level
        score_buckets = {'300-500': {'total': 0, 'high': 0}, '501-650': {'total': 0, 'high': 0}, '651-750': {'total': 0, 'high': 0}, '751-850': {'total': 0, 'high': 0}}
        for a in applicants:
            cs = a.credit_score or 0
            if cs <= 500:
                key = '300-500'
            elif cs <= 650:
                key = '501-650'
            elif cs <= 750:
                key = '651-750'
            else:
                key = '751-850'
            score_buckets[key]['total'] += 1
            if a.risk_level == 'HIGH':
                score_buckets[key]['high'] += 1
        
        credit_labels = list(score_buckets.keys())
        credit_data = [round((b['high'] / b['total']) * 100, 1) if b['total'] > 0 else 0 for b in score_buckets.values()]

        # 3. Age vs Default Rate
        age_buckets = {'18-30': {'total': 0, 'high': 0}, '31-45': {'total': 0, 'high': 0}, '46-60': {'total': 0, 'high': 0}, '60+': {'total': 0, 'high': 0}}
        for a in applicants:
            age = a.age or 0
            if age <= 30:
                key = '18-30'
            elif age <= 45:
                key = '31-45'
            elif age <= 60:
                key = '46-60'
            else:
                key = '60+'
            age_buckets[key]['total'] += 1
            if a.risk_level == 'HIGH':
                age_buckets[key]['high'] += 1
        
        age_labels = list(age_buckets.keys())
        age_data = [round((b['high'] / b['total']) * 100, 1) if b['total'] > 0 else 0 for b in age_buckets.values()]

        return {
            "income_dist": {"labels": list(brackets.keys()), "data": list(brackets.values())},
            "credit_default": {"labels": credit_labels, "data": credit_data},
            "age_default": {"labels": age_labels, "data": age_data}
        }

    def get_history(self, days: int = None) -> list:
        """Get application history, optionally filtered by time."""
        query = self._db.query(Applicant).filter(Applicant.user_id == self._user_id).order_by(Applicant.created_at.desc())
        
        if days:
            cutoff = datetime.utcnow() - timedelta(days=days)
            query = query.filter(Applicant.created_at >= cutoff)
        
        applicants = query.all()
        return [{
            "id": a.id,
            "name": a.name,
            "age": a.age,
            "gender": a.gender,
            "income": f"₹{a.income:,.0f}" if a.income else "N/A",
            "loan_amount": f"₹{a.loan_amount:,.0f}" if a.loan_amount else "N/A",
            "credit_score": int(a.credit_score) if a.credit_score else "N/A",
            "risk_score": f"{a.risk_score:.1f}%" if a.risk_score else "N/A",
            "risk_level": a.risk_level,
            "recommendation": a.recommendation,
            "date": a.created_at.strftime("%d %b %Y, %I:%M %p") if a.created_at else "N/A"
        } for a in applicants]

    def get_applicant_detail(self, applicant_id: int) -> dict:
        """Get full details including XAI reasons for a specific applicant."""
        a = self._db.query(Applicant).filter(Applicant.id == applicant_id, Applicant.user_id == self._user_id).first()
        if not a:
            return None
        return {
            "name": a.name,
            "risk_score": a.risk_score,
            "risk_level": a.risk_level,
            "recommendation": a.recommendation,
            "reasons": a.reasons or ["No explanation available."]
        }
