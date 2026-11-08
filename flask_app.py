import os
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from src.db.database import SessionLocal
from src.db import crud
from src.services.auth_service import authenticate_user, init_db_and_auth
from src.services.data_service import DataService
from src.services.interactive_risk_service import InteractiveRiskService

app = Flask(__name__, static_folder='assets', static_url_path='/assets', template_folder='templates')
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-finsight-key-2026")

# Initialize DB on startup
with app.app_context():
    init_db_and_auth()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# --- Page Routes ---

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('email', '').strip()
        password = request.form.get('password', '')
            
        db = SessionLocal()
        try:
            user = authenticate_user(db, username, password)
            if user:
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['role'] = user['role']
                session['full_name'] = user['full_name']
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials. Please try again.', 'error')
        finally:
            db.close()
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        full_name = request.form.get('full_name', '').strip()
        
        if not username or not email or not password or not full_name:
            flash('All fields are required.', 'error')
            return render_template('register.html')
            
        db = SessionLocal()
        try:
            # Check if user exists by username
            existing = crud.get_user_by_username(db, username)
            if existing:
                flash('Username already exists.', 'error')
                return render_template('register.html')
            
            # Check if email already exists
            existing_email = crud.get_user_by_username_or_email(db, email)
            if existing_email:
                flash('An account with this email already exists.', 'error')
                return render_template('register.html')
                
            # Create user
            crud.create_user(db, username, email, password, "User", full_name)
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except ValueError as e:
            flash(str(e), 'error')
            return render_template('register.html')
        except Exception as e:
            flash('An error occurred during registration. Please try again.', 'error')
            return render_template('register.html')
        finally:
            db.close()
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard/executive.html', user=session)

@app.route('/dashboard/risk-assessment')
@login_required
def risk_assessment():
    return redirect(url_for('predict'))

@app.route('/dashboard/analytics')
@login_required
def analytics():
    return render_template('dashboard/analytics.html', user=session)

@app.route('/dashboard/segmentation')
@login_required
def segmentation():
    return render_template('dashboard/segmentation.html', user=session)

@app.route('/dashboard/predict')
@login_required
def predict():
    return render_template('dashboard/predict.html', user=session)

@app.route('/dashboard/compare')
@login_required
def compare():
    return render_template('dashboard/compare.html', user=session)

@app.route('/dashboard/history')
@login_required
def history():
    return render_template('dashboard/history.html', user=session)

# --- API Routes (All Database-Driven) ---

@app.route('/api/dashboard/kpis')
@login_required
def api_kpis():
    db = SessionLocal()
    try:
        ds = DataService(db, session['user_id'])
        return jsonify(ds.get_dashboard_kpis())
    finally:
        db.close()

@app.route('/api/analytics')
@login_required
def api_analytics():
    db = SessionLocal()
    try:
        ds = DataService(db, session['user_id'])
        return jsonify(ds.get_analytics_data())
    finally:
        db.close()

@app.route('/api/history')
@login_required
def api_history():
    days = request.args.get('days', None, type=int)
    db = SessionLocal()
    try:
        ds = DataService(db, session['user_id'])
        return jsonify(ds.get_history(days))
    finally:
        db.close()

@app.route('/api/history/<int:app_id>')
@login_required
def api_history_detail(app_id):
    db = SessionLocal()
    try:
        ds = DataService(db, session['user_id'])
        detail = ds.get_applicant_detail(app_id)
        if not detail:
            return jsonify({"error": "Not found"}), 404
        return jsonify(detail)
    finally:
        db.close()

@app.route('/api/predict', methods=['POST'])
@login_required
def api_interactive_predict():
    data = request.json
    
    rs = InteractiveRiskService()
    assessment = rs.assess_new_application(data)
    
    if "error" in assessment:
        return jsonify({"error": assessment["error"]}), 500

    # Save the application to the database
    db = SessionLocal()
    try:
        rs.save_application(db, data, assessment, session['user_id'])
    finally:
        db.close()
        
    return jsonify(assessment)

@app.route('/api/compare', methods=['POST'])
@login_required
def api_compare():
    data = request.json
    cust_a = data.get('customer_a', {})
    cust_b = data.get('customer_b', {})
    
    rs = InteractiveRiskService()
    res_a = rs.assess_new_application(cust_a)
    res_b = rs.assess_new_application(cust_b)
    
    return jsonify({
        "customer_a_result": res_a,
        "customer_b_result": res_b
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
