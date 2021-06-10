API_LINK = "http://127.0.0.1:5000/"
# You'd want to create JWT auth for accessing this API
# which will be given to the user in our webapp and also refreshed at a regular time interval
# the architecture is a microservice architecture so the data layer needs to be separate from the data layer from the original instance supporting the actual webapp
# this new data layer will have access to the authentication code as well as the init_info from Training, which means when training is complete, node needs to invoke a push to
# the new database which is specific to the APIs
# I suppose the authentication code will be used by the user here in this script to be called in the other script, you can define how it will flow.
