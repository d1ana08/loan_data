from fastapi import APIRouter
from pydantic import BaseModel
import joblib

model = joblib.load('log_model.pkl')
scaler = joblib.load('scaler.pkl')

predict_router = APIRouter(prefix='/predict', tags=['Predict'])


class LoanPredictSchema(BaseModel):
    person_age: int
    person_income: int
    person_emp_exp: int
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: int
    credit_score: int
    person_gender: str
    person_education: str
    person_home_ownership: str
    loan_intent: str
    previous_loan_defaults_on_file: str


@predict_router.post('/')
async def predict(loan: LoanPredictSchema):
    new_loan = loan.dict()

    gender = new_loan.pop('person_gender')
    education = new_loan.pop('person_education')
    home = new_loan.pop('person_home_ownership')
    intent = new_loan.pop('loan_intent')
    defaults = new_loan.pop('previous_loan_defaults_on_file')

    new = [
        new_loan['person_age'],
        new_loan['person_income'],
        new_loan['person_emp_exp'],
        new_loan['loan_amnt'],
        new_loan['loan_int_rate'],
        new_loan['loan_percent_income'],
        new_loan['cb_person_cred_hist_length'],
        new_loan['credit_score'],
    ]

    gender = [
        1 if gender == 'male' else 0,
    ]

    education = [
        1 if education == 'Bachelor' else 0,
        1 if education == 'Doctorate' else 0,
        1 if education == 'High School' else 0,
        1 if education == 'Master' else 0,
    ]

    home = [
        1 if home == 'OTHER' else 0,
        1 if home == 'OWN' else 0,
        1 if home == 'RENT' else 0,
    ]

    intent = [
        1 if intent == 'EDUCATION' else 0,
        1 if intent == 'HOMEIMPROVEMENT' else 0,
        1 if intent == 'MEDICAL' else 0,
        1 if intent == 'PERSONAL' else 0,
        1 if intent == 'VENTURE' else 0,
    ]

    defaults = [
        1 if defaults == 'Yes' else 0,
    ]

    features = new + gender + education + home + intent + defaults

    neew = scaler.transform([features])
    loan = float(model.predict_proba(neew)[0][1])
    approved = loan >= 0.5

    return {
        "approved": approved,
        "probability": round(loan, 2)
    }
