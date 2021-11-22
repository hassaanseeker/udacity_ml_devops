docker build . -t churn_predictor
docker run churn_predictor 

docker cp churn_predictor:/ChurnPredictor/project/images/results/ project/images