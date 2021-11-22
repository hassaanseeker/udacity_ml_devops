### Running the churn predictor code

Follow the below guidelines

##### Starting Docker Container.

While inside the project_1 folder run the below command below command to create a docker image.

```docker build . -t churn_predictor```

##### Running Docker Container.

Still inside the project_1 folder run the image using the below command.

```docker run churn_predictor```

##### Copy results from Docker to host.

When the command prompt displays All results saved inisde the project_1 folder run the below command to copy the results from the docker container.

```docker cp churn_predictor:/ChurnPredictor/project/images/results/ project/images```