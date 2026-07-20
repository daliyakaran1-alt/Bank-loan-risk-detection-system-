from src.services.risk_service import RiskService
from src.services.data_service import DataService
ds = DataService()
rs = RiskService()
for cid in [100002, 100003, 100004, 100006, 100007]:
    cust = ds.get_customer_data(cid)
    result = rs.assess_loan_application(cust)
    print(f"ID:{cid}  Score:{result['risk_score']}  Rec:{result['recommendation']}")
