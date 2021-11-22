### Running the churn predictor code

Follow the below guidelines

##### Starting Docker Container.

While inside the project_1 folder run the below command below command to create a docker image.

'docker build . -t churn_predictor'

Second run the image using the below.
docker run churn_predictor 

When the command prompt displays All results saved. Run the below command to copy the results from the docker container.

docker cp churn_predictor:/ChurnPredictor/project/images/results/ project/images