# Detect Handguns Demo

This is a demo project for demonstrating how to apply machine learning for object detection and then create and API backend to request detection of handguns in images.

## Running Locally

- Install docker and docker-compose on your PC
  - This was written with docker desktop 4.25.1
- Open a terminal at the base of this project and run the following:

 ```bash
 docker-compose -f ./docker/object_detection_demo/docker-compose.local.yml build
 docker-compose -f ./docker/object_detection_demo/docker-compose.local.yml up
 ```

- If you need to enter the app docker container run the following in another terminal:

```bash
docker exec -it objection_detection_demo_app_local bash
```

### Run Unit Tests

- After having the container up an running, open a new terminal and run the following:

```bash
docker exec -it objection_detection_demo_app_local python3 -m coverage run -m pytest
docker exec -it objection_detection_demo_app_local python3 -m coverage report
```
