# docker note

# CHECK

\# see the available container \
$ docker images \
\# check what containers are running \
$ docker ps \
\# check all containers(exit and running) \
$ docker ps -a

# RUN and EXIT and REMOVE
\# using this can run multiple commands within the virtual machine \
$ docker run -it <NAME> \
\# to give argument [for mac] \
$ docker run --name experiment2 --rm -v $(pwd)<Yourpath1>:<Virtualpath1> -v $(pwd)<Yourpath2>:<Virtualpath2> predict:latest <Virtualpath1> <Virtualpath2> \
\# to leave the the container \
/# exit \
\# to kill specific container \
$ docker rm <container_id>

# BUILD
\# if stay in the parent directory \
$ docker build -f dockerfile/trainer.dockerfile . -t trainer:latest \
\# if inside dockerfile directory \
$ docker build -f trainer.dockerfile ../ -t trainer:latest
