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


@predict_router.post('/predict/')
async def chek_loan_account(loan: LoanPredictSchema):
    loan_dict = loan.dict()

    gender = loan_dict.pop('person_gender')
    gender_1_0 = [
        1 if gender == 'male' else 0
    ]

    education = loan_dict.pop('person_education')
    education_1_0 = [
        1 if education == 'Bachelor' else 0,
        1 if education == 'Doctorate' else 0,
        1 if education == 'High School' else 0,
        1 if education == 'Master' else 0,
    ]

    home = loan_dict.pop('person_home_ownership')
    home_1_0 = [
        1 if home == 'OTHER' else 0,
        1 if home == 'OWN' else 0,
        1 if home == 'RENT' else 0,
    ]


    intent = loan_dict.pop('loan_intent')
    intent_1_0 = [
        1 if intent == 'EDUCATION' else 0,
        1 if intent == 'HOMEIMPROVEMENT' else 0,
        1 if intent == 'MEDICAL' else 0,
        1 if intent == 'PERSONAL' else 0,
        1 if intent == 'VENTURE' else 0,
    ]

    defaults = loan_dict.pop('previous_loan_defaults_on_file')
    defaults1_0 = [
        1 if defaults == 'Yes' else 0,
    ]


    loan_data = list(loan_dict.values()) +  gender_1_0 + education_1_0 + home_1_0 + intent_1_0 + defaults1_0

    scaled_data = scaler.transform([loan_data])
    pred = model.predict(scaled_data)[0]
    final_pred = "Approved" if int(pred) == 1 else 'Rejected'
    return {"answer": final_pred}