from fastapi import FastAPI
import uvicorn
from loan_app.api import predict

loan_app = FastAPI()
loan_app.include_router(predict.predict_router)



if __name__ == '__main__':
    uvicorn.run(loan_app, host='127.0.0.1', port=8000)